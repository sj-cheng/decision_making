import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from solvers.e50_distributed_runtime import LocalPlan
from watch_e52_experiments import alive, inspect
from run_e52_train_nar_nae import records_from_run


class MonitorAndTerminalTests(unittest.TestCase):
    def test_missing_pid_is_not_running(self):
        self.assertTrue(alive(os.getpid(),'python'))
        self.assertFalse(alive(999999999,'python'))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'status.json').write_text(json.dumps(dict(supervisor_pid=999999999,stages={'pipeline':dict(state='failed')})))
            result=inspect(root)
            self.assertFalse(result['supervisor_alive'])
            self.assertTrue(any('pipeline failed' in a for a in result['alerts']))

    def test_episode_stops_after_both_targets_captured(self):
        from run_e50_distributed_dynamic_smoke import run
        class Stay:
            def act(self,obs):return LocalPlan(obs.agent_id,obs.pose,np.zeros(2),0.,1,0)
        with tempfile.TemporaryDirectory() as folder:
            case=Path(folder)/'case.json';out=Path(folder)/'run.json'
            case.write_text(json.dumps(dict(map=[],worlds=[dict(initial=[0.,0.,5.,5.,0.,0.,5.,5.])],horizon=80)))
            with patch('run_e50_distributed_dynamic_smoke.make_local_geometry_mcts',return_value=Stay()):
                result=run(case,out,cycles=160)
            self.assertEqual(len(result['captures']),2)
            self.assertTrue(result['terminal'])
            self.assertEqual(result['cycles'],1)
            self.assertEqual(result['requested_cycles'],160)
            progress=json.loads(out.with_suffix('.progress.json').read_text())
            self.assertEqual(progress['state'],'complete')
            self.assertEqual(progress['completed_cycles'],1)

    def test_terminal_short_return_is_valid_but_nonterminal_is_not(self):
        root=[dict(features=[0.]*14,visits=2)]
        row=dict(cycle=0,search={a:dict(root=root) for a in ('0','1')},plans={a:dict(candidate_id=0) for a in ('0','1')},
                 captures=[dict(target=0,cycle=0),dict(target=1,cycle=0)],
                 sensor_stats={a:dict(visible_targets=0) for a in ('0','1')},beliefs={},pursuers=[[0.,0.],[1.,0.]])
        records=records_from_run(dict(logs=[row]))
        self.assertEqual(len(records),2)
        self.assertGreater(records[0]['returns'][0],7.)
        row['captures']=[]
        self.assertEqual(records_from_run(dict(logs=[row])),[])


if __name__=='__main__':unittest.main()
