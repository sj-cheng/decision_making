"""A saved training phase alone must never imply a live process."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import watch_search_rl_reward as monitor


class RewardMonitorTests(unittest.TestCase):
    def test_stale_phase_reports_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'status.json').write_text(json.dumps({'phase':'longer_paired_training'}))
            with patch.object(monitor,'running_pids',return_value=[]):
                self.assertIn('已停止：以下为旧进度',monitor.render(root))

    def test_actual_pid_reports_running(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(monitor,'running_pids',return_value=[1234]):
                self.assertIn('运行中 PID 1234',monitor.render(Path(directory)))

    def test_completed_job_needs_no_live_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'status.json').write_text(json.dumps({'phase':'complete'}))
            with patch.object(monitor,'running_pids',return_value=[]):
                self.assertIn('主任务：已完成',monitor.render(root))


if __name__=='__main__':unittest.main()
