"""Saved l45 PPO progress must never be confused with a live driver."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import watch_l45_ppo as monitor


class L45PPOMonitorTests(unittest.TestCase):
    def make_process(self, proc, pid, argv, cwd, state='S'):
        entry = proc / str(pid)
        entry.mkdir()
        (entry / 'stat').write_text(f'{pid} (python) {state} 1 2 3\n')
        (entry / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())
        (entry / 'cwd').symlink_to(cwd, target_is_directory=True)

    def test_stale_training_state_without_process_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'status.json').write_text(json.dumps({
                'phase': 'training', 'pid': 456789, 'seed': 0, 'episode': 64,
            }))
            with patch.object(monitor, 'running_pids', return_value=[]):
                display = monitor.render(output)
            self.assertIn('状态：已中断/停止', display)
            self.assertNotIn('状态：运行中', display)

    def test_no_saved_status_is_unstarted(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(monitor, 'running_pids', return_value=[]):
                display = monitor.render(Path(directory))
            self.assertIn('状态：未启动', display)

    def test_actual_runner_requires_exact_script_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / 'proc'
            proc.mkdir()
            output = root / 'experiment'
            output.mkdir()
            command = ['/env/python3.11', '-u', str(monitor.RUNNER)]
            self.make_process(proc, 101, command + ['--output', str(output)], root)
            self.make_process(proc, 102, command + ['--output=' + str(root / 'other')], root)
            self.make_process(proc, 103, ['/env/python', str(root / 'train_l45_ppo.py'),
                                        '--output', str(output)], root)
            self.make_process(proc, 104, ['/bin/bash', str(monitor.RUNNER),
                                        '--output', str(output)], root)
            self.assertEqual(monitor.running_pids(output, proc), [101])

    def test_spawn_worker_and_zombie_are_not_live_runners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / 'proc'
            proc.mkdir()
            command = ['/env/python', str(monitor.RUNNER), '--output', str(root)]
            self.make_process(proc, 201, command, root, state='Z')
            self.make_process(proc, 202, ['/env/python', '-c',
                                        'from multiprocessing.spawn import spawn_main',
                                        str(monitor.RUNNER), '--output', str(root)], root)
            self.make_process(proc, 203, ['/env/python', '-m', 'multiprocessing.spawn',
                                        str(monitor.RUNNER), '--output', str(root)], root)
            self.assertEqual(monitor.running_pids(root, proc), [])

    def test_default_and_relative_output_paths(self):
        cwd = monitor.CODE.parent
        self.assertEqual(monitor.runner_output(
            ['/env/python', '-u', 'code/train_l45_ppo.py'], cwd), monitor.DEFAULT_OUTPUT)
        self.assertEqual(monitor.runner_output(
            ['/env/python', str(monitor.RUNNER), '--output', 'relative/output'], cwd),
            (cwd / 'relative/output').resolve())
        self.assertEqual(monitor.runner_output(
            ['/env/python', str(monitor.RUNNER), '--output=relative/output'], cwd),
            (cwd / 'relative/output').resolve())

    def test_completion_requires_report_and_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'status.json').write_text(json.dumps({'phase': 'complete'}))
            with patch.object(monitor, 'running_pids', return_value=[]):
                self.assertNotIn('状态：已完成', monitor.render(output))
                (output / 'REPORT.md').write_text('# Report\n')
                self.assertNotIn('状态：已完成', monitor.render(output))
                (output / 'results.json').write_text('{}\n')
                self.assertIn('状态：已完成', monitor.render(output))

    def test_live_driver_is_reported_and_eta_excludes_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'experiment.json').write_text(json.dumps({
                'config': {'seeds': [0], 'episodes': 128, 'batch': 64},
            }))
            (output / 'seed0').mkdir()
            (output / 'seed0/progress.json').write_text(json.dumps({
                'episode': 64, 'total': 128, 'sampling_seconds': 120,
                'update_seconds': 2, 'best_episode': 0,
            }))
            with patch.object(monitor, 'running_pids', return_value=[1234]):
                display = monitor.render(output)
            self.assertIn('状态：运行中，PID 1234', display)
            self.assertIn('64/128 局', display)
            self.assertIn('最近一批 64 局', display)
            self.assertIn('不含后续验证、完整搜索确认、回归和汇总', display)


if __name__ == '__main__':
    unittest.main()
