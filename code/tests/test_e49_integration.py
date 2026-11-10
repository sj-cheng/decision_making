"""Tests for the independent E49 sampling review and frozen engineering matrix."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from e49_rules import create_environment
from e49_engineering import paired_sampling,prepare_engineering,_runner_functions
from e49_audit import audit_frame,diagnostics
from run_e49_benchmark import observation


def payload(pursuers,targets,action,enemy):
    env=create_environment([]);s=np.zeros((53,1));s[:8,0]=np.r_[np.ravel(targets),np.ravel(pursuers)];s[9:13]=1
    n=env.step(s,np.r_[np.ravel(enemy),np.ravel(action)][:,None])
    frame=dict(state=s[:,0].tolist(),next_state=n[:,0].tolist(),action=np.asarray(action).tolist(),opponent_action=np.asarray(enemy).tolist(),observation=observation(s,np.empty((0,2,2)),40))
    frame['capture_events']=audit_frame(frame,np.empty((0,2,2)))
    return dict(trace=[frame],summary=diagnostics([frame],40))


class E49IntegrationTests(unittest.TestCase):
    def test_sampling_reports_small_time_shift(self):
        p=payload([[0,0],[-7,-7]],[[1.2,0],[7,7]],[[1.,0],[0,0]],[[-1.5,0],[0,0]])
        result=paired_sampling(p,[])
        self.assertTrue(result['passed'])
        self.assertLessEqual(abs(result['time_differences'][0]['difference']),.005+1e-10)
        self.assertLessEqual(result['maximum_position_difference'],.02)

    def test_sampling_blocks_a_missed_short_encounter(self):
        p=payload([[-.005,.999999],[-7,-7]],[[0,0],[7,7]],[[1.,0],[0,0]],[[0,0],[0,0]])
        self.assertFalse(p['summary']['at_least_one'])
        result=paired_sampling(p,[])
        self.assertFalse(result['passed'])
        self.assertTrue(any(v['reason']=='captured_set_changed' for v in result['mismatches']))

    def test_core_fingerprint_excludes_cli_changes(self):
        source='def observation(): return 1\ndef reset_history(): return 2\ndef run_episode(): return 3\n'
        self.assertEqual(_runner_functions(source+'def main(): return 4\n'),
                         _runner_functions(source+'def main(): return 5\n'))
        self.assertNotEqual(_runner_functions(source),_runner_functions(source.replace('return 3','return 6')))

    def test_frozen_engineering_matrix_is_separate_and_complete(self):
        with tempfile.TemporaryDirectory() as root:
            record=prepare_engineering(Path(root))
            self.assertEqual(len(record['entries']),21)
            self.assertEqual(len(record['tasks']),66)
            self.assertEqual(len({t['name'] for t in record['tasks']}),22)
            for tier in ('sparse','16','24'):
                tasks=[t for t in record['tasks'] if t['obstacle_tier']==tier]
                self.assertEqual(len(tasks),22)
                self.assertEqual(len({t['family'] for t in tasks}),7)
                self.assertTrue(all(t['horizon']==40 and t['opponent_simulations']==3000 for t in tasks))
            self.assertTrue(all('/engineering/cases/' in e['case'] for e in record['entries']))


if __name__=='__main__':unittest.main()
