"""The final-search monitor distinguishes real drivers from saved progress."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import watch_l45_final_search as monitor


class FinalSearchMonitorTests(unittest.TestCase):
    def make_process(self, proc, pid, argv, cwd, state='S'):
        entry = proc / str(pid)
        entry.mkdir()
        (entry / 'stat').write_text(f'{pid} (python) {state} 1 2 3\n')
        (entry / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())
        (entry / 'cwd').symlink_to(cwd, target_is_directory=True)

    def test_old_status_pid_is_not_liveness(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'status.json').write_text(json.dumps({
                'phase': 'evaluation', 'pid': 1234, 'evaluation': 'confirmation',
                'completed': 42, 'total': 100,
            }))
            with patch.object(monitor, 'running_pids', return_value=[]):
                display = monitor.render(output)
            self.assertIn('状态：已中断/停止', display)
            self.assertNotIn('状态：运行中', display)
            self.assertIn('本实验不训练', display)

    def test_empty_directory_is_unstarted(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(monitor, 'running_pids', return_value=[]):
                self.assertIn('状态：未启动', monitor.render(directory))

    def test_completion_requires_phase_and_both_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'status.json').write_text('{"phase": "complete"}')
            with patch.object(monitor, 'running_pids', return_value=[]):
                self.assertNotIn('已完成', monitor.render(output))
                (output / 'REPORT.md').write_text('# Report\n')
                self.assertNotIn('已完成', monitor.render(output))
                (output / 'results.json').write_text('{}\n')
                self.assertIn('状态：已完成', monitor.render(output))
                (output / 'status.json').write_text('{"phase": "summarizing"}')
                self.assertNotIn('状态：已完成', monitor.render(output))

    def test_only_exact_runner_and_output_are_live(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / 'proc'
            proc.mkdir()
            command = ['/env/python3.11', '-u', str(monitor.RUNNER)]
            self.make_process(proc, 101, command + ['--output', str(root)], root)
            self.make_process(proc, 102, command + ['--output=' + str(root / 'other')], root)
            self.make_process(proc, 103, ['/env/python', str(root / monitor.RUNNER.name),
                                        '--output', str(root)], root)
            self.make_process(proc, 104, ['/bin/bash', str(monitor.RUNNER),
                                        '--output', str(root)], root)
            self.assertEqual(monitor.running_pids(root, proc), [101])

    def test_spawn_workers_and_zombies_do_not_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / 'proc'
            proc.mkdir()
            command = ['/env/python', str(monitor.RUNNER), '--output', str(root)]
            self.make_process(proc, 201, command, root, state='Z')
            self.make_process(proc, 202, ['/env/python', '-c', 'spawn_main()',
                                        str(monitor.RUNNER), '--output', str(root)], root)
            self.make_process(proc, 203, ['/env/python', '-m', 'multiprocessing.spawn',
                                        str(monitor.RUNNER), '--output', str(root)], root)
            self.assertEqual(monitor.running_pids(root, proc), [])

    def test_relative_and_default_output_resolution(self):
        cwd = monitor.CODE.parent
        self.assertEqual(monitor.runner_output(
            ['/env/python', '-u', 'code/' + monitor.RUNNER.name], cwd), monitor.DEFAULT_OUTPUT)
        self.assertEqual(monitor.runner_output(
            ['/env/python', str(monitor.RUNNER), '--output=relative/result'], cwd),
            (cwd / 'relative/result').resolve())
        self.assertIsNone(monitor.runner_output(
            ['/env/python', str(monitor.RUNNER), '--output'], cwd))

    def test_workers_read_actual_commandline_and_default_80(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / 'proc'
            proc.mkdir()
            command = ['/env/python', str(monitor.RUNNER)]
            self.make_process(proc, 301, command, root)
            self.make_process(proc, 302, command + ['--workers', '48'], root)
            self.make_process(proc, 303, command + ['--workers=12'], root)
            self.assertEqual(monitor.configured_workers(301, proc), 80)
            self.assertEqual(monitor.configured_workers(302, proc), 48)
            self.assertEqual(monitor.configured_workers(303, proc), 12)

    def test_cache_count_comes_from_manifest_and_real_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            folder = output / 'confirmation'
            folder.mkdir()
            (folder / 'manifest.json').write_text(json.dumps([
                {'key': 'a'}, {'key': 'b'}, {'key': 'c'},
            ]))
            (folder / 'a.json').write_text('{}')
            (folder / 'unexpected.json').write_text('{}')
            (output / 'status.json').write_text(json.dumps({
                'phase': 'evaluation', 'evaluation': 'confirmation',
                'completed': 99, 'total': 99,
            }))
            with patch.object(monitor, 'running_pids', return_value=[]):
                display = monitor.render(output)
            self.assertIn('四障碍独立确认缓存实际文件：1/3', display)
            self.assertIn('保存计数：99/99', display)
            self.assertEqual(monitor.cached_progress(output, 'confirmation'), (1, 3))

    def test_live_pid_mismatch_and_eta_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            folder = output / 'regression'
            folder.mkdir()
            (folder / 'manifest.json').write_text('[{"key":"one"},{"key":"two"}]')
            (folder / 'one.json').write_text('{}')
            (output / 'status.json').write_text(json.dumps({
                'phase': 'evaluation', 'evaluation': 'regression', 'pid': 900,
                'throughput': {'elapsed_seconds': 60, 'rate': 1 / 60},
            }))
            with patch.object(monitor, 'running_pids', return_value=[901]), \
                    patch.object(monitor, 'configured_workers', return_value=80):
                display = monitor.render(output)
            self.assertIn('状态：运行中，PID 901', display)
            self.assertIn('实际评测并行数：PID 901: 80', display)
            self.assertIn('保存状态的 PID 与当前进程不同', display)
            self.assertIn('当前 regression 剩余：1 分钟', display)
            self.assertIn('不含后续评测、诊断与汇总', display)

    def test_missing_throughput_never_guesses_eta(self):
        self.assertIsNone(monitor.split_eta({}, 0, 100))
        self.assertIsNone(monitor.split_eta({'rate': 10}, 0, 100))
        self.assertIsNone(monitor.split_eta({'rate': 0, 'elapsed_seconds': 1}, 0, 100))
        self.assertEqual(monitor.split_eta({'rate': 1, 'elapsed_seconds': 10}, 0, 120),
                         '2 分钟')


if __name__ == '__main__':
    unittest.main()
