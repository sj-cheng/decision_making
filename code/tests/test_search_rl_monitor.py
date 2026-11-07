"""Read-only progress must distinguish episode counts and success metrics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from watch_search_rl import read_metrics, snapshot, render


class MonitorTests(unittest.TestCase):
    def test_inflight_jsonl_and_role_specific_eta(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'invocation.json').write_text(json.dumps(dict(episodes=100, workers=32, episodes_per_update=20)))
            (out/'fork.json').write_text(json.dumps(dict(source_counters=[40,40])))
            rows = [dict(role=0, episodes=[60,40], seconds=20), dict(role=1, episodes=[60,60], seconds=40)]
            (out/'train_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"role":0')
            data = snapshot(out, dict(running=True,pids=[123],workers=32,busy=3,cpu=300))
            self.assertEqual(len(read_metrics(out/'train_metrics.jsonl')), 2)
            self.assertEqual(data['counters'], [60,60])
            self.assertEqual(data['eta'], 40*1+40*2)
            self.assertEqual(data['next_role'], 0)
            self.assertIn('60.0%', render(data, width=84, height=22))
            self.assertLessEqual(len(render(data, width=84, height=22).splitlines()), 21)

    def test_validation_uses_legal_survival_and_double_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            episodes = [dict(legal_survival=0., double_capture=1.), dict(legal_survival=0., double_capture=0.)]
            for role in range(2):
                (out/f'validation_role{role}_ep0.json').write_text(json.dumps(dict(score=0.,episodes=episodes)))
            data = snapshot(out, dict(running=False,pids=[],workers=0,busy=None,cpu=None))
            self.assertEqual(data['validation'][0]['baseline']['rate'], 0.)
            self.assertEqual(data['validation'][1]['baseline']['rate'], .5)
            (out/'ppo_final.pt').touch()
            data = snapshot(out, data['activity'])
            self.assertEqual(data['phase'], 'export')
            (out/'export').mkdir(); (out/'export/policy.pt').touch()
            data = snapshot(out, data['activity'])
            self.assertEqual(data['phase'], 'evaluation')
            self.assertNotEqual(data['phase'], 'complete')

    def test_baseline_survives_multiple_continuation_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline, previous, current = [root/name for name in ('baseline','previous','current')]
            for directory in (baseline, previous, current):directory.mkdir()
            (previous/'fork.json').write_text(json.dumps(dict(source=str(baseline/'ppo_latest.pt'))))
            (current/'fork.json').write_text(json.dumps(dict(source=str(previous/'ppo_latest.pt'),source_counters=[2000,2000])))
            for role in range(2):
                data=dict(score=.5,episodes=[dict(legal_survival=.25,double_capture=.5)])
                (baseline/f'validation_role{role}_ep0.json').write_text(json.dumps(data))
                (previous/f'validation_role{role}_ep2000.json').write_text(json.dumps(data))
            data=snapshot(current,dict(running=True,pids=[123],workers=32,busy=8,cpu=800))
            self.assertEqual(data['validation'][0]['baseline']['rate'],.25)
            self.assertEqual(data['validation'][1]['baseline']['rate'],.5)
            self.assertEqual(data['validation'][0]['at'],2000)


if __name__ == '__main__':
    unittest.main()
