"""Checks for repeat fingerprints and controlled model-evaluation semantics."""
import copy,json,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repeat_frozen_d_closed_loop import SOURCE,scientific_trace
from diagnose_frozen_d_root_rollout import evaluate,target_paths


class RepeatDiagnosisTests(unittest.TestCase):
    def test_fingerprint_ignores_only_timing_and_retains_actions_and_values(self):
        rows=json.loads((SOURCE/'confirmation/results.json').read_text())
        trace=json.loads(Path(rows[0]['episode']).read_text())['trace'][:1]
        changed=copy.deepcopy(trace)
        changed[0]['own_ms']+=123.
        changed[0]['statistics']['total_ms']+=123.
        self.assertEqual(scientific_trace(trace),scientific_trace(changed))
        changed[0]['own_action'][0]+=.1
        self.assertNotEqual(scientific_trace(trace),scientific_trace(changed))

    def test_discounted_visibility_then_capture_and_horizon_prefix(self):
        own=np.zeros((25,2));own[:,0]=np.minimum(np.arange(25),8.)
        targets=np.zeros((2,25,2));targets[:,:,0]=8.
        short=evaluate(own,targets,np.zeros((0,2,2)),8)
        long=evaluate(own,targets,np.zeros((0,2,2)),24)
        # First visible at step2 (distance6), capture at step8 (distance0).
        np.testing.assert_allclose(short[:,2],.95+.95**7)
        np.testing.assert_array_equal(short,long)
        np.testing.assert_array_equal(short[:,:2],np.ones((2,2)))

    def test_draws_preserve_importance_weights_and_common_target_prefix(self):
        points=np.array([[0.,0.],[8.,0.]])
        links=np.zeros((2,49),np.int32);links[1,:]=1
        lengths=np.ones(2,np.int32)
        probabilities=np.zeros((2,49,9));probabilities[:,0,:]=1.
        ids=np.ones(64,np.int64);ids[0]=0
        weights=np.full(64,.1/63);weights[0]=.9
        modes=np.zeros((64,9));modes[:,0]=1.
        paths=target_paths(points,links,lengths,probabilities,ids,weights,modes,10000,24,17)
        self.assertLess(abs(np.mean(paths[:,0,0]==0)-.9),.015)
        np.testing.assert_array_equal(paths[:,:9],np.repeat(paths[:,0:1],9,axis=1))


if __name__=='__main__':unittest.main()
