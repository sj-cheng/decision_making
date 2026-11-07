"""New-map restoration, frozen method identity, matching and resumable records."""
import copy
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import run_l45_direction_validation as m

class DirectionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg=dict(O=dict(models=str(m.SOURCE/'models/O'),model_hash=m.o.model_fingerprint(m.SOURCE/'models/O')),
            L=dict(models=str(m.old.SOURCE/'models/L'),model_hash=m.o.model_fingerprint(m.old.SOURCE/'models/L')))

    def task(self,uid='G',map_id='offset_corridor',opponent='Search Only',late=True):
        spec=m.make_scenario(58261024,map_id)
        if late:
            state=np.array(spec['initial_state']);p,_=m.restore_scenario(spec);state[p.time_idx,0]=77.
            spec['initial_state']=state.tolist();spec['episode_hash']=m.o.json_hash({k:v for k,v in spec.items() if k!='episode_hash'})
        args=m.parse_args(['--smoke']);behavior=m.effective(uid,self.reg)
        return dict(key=uid+'-'+map_id+'-'+opponent,uid=uid,spec=spec,registry=self.reg,effective=behavior,
            role=1,opponent=opponent,timing_context='parallel',mode='search',policy_id=uid,
            own_settings=m.settings(args),opponent_settings=m.settings(args,0))

    def test_maps_restore_limits_encoding_and_transition(self):
        for map_id in m.MAPS:
            task=self.task(map_id=map_id);p,state,own,other=m.make_pair(task)
            self.assertEqual([x.tolist() for x in p.obstacles],m.MAPS[map_id])
            np.testing.assert_array_equal(p.action_lims,np.array(task['spec']['action_lims']))
            wr=own.solver._problem_wrapper_for(p)
            for r in range(4):np.testing.assert_allclose(np.asarray(wr.policy_encoding(state,r)).reshape(-1),p.policy_encoding(state,r).reshape(-1),atol=3e-5,rtol=0)
            action=np.zeros((8,1));action[:2,0]=[.2,-.3]
            np.testing.assert_allclose(np.asarray(wr.step(state,action,1.)).reshape(-1),p.step(state,action,1.).reshape(-1),atol=3e-5,rtol=0)
            memory=state.copy();p._update_observation_memory(memory,memory,0.)
            np.testing.assert_array_equal(state,memory)

    def test_physical_hash_ignores_map_and_memory(self):
        a=m.make_scenario(58261025,'none');b=copy.deepcopy(a);b['map_id']='offset_corridor';b['obstacles']=m.MAPS['offset_corridor']
        b['initial_state'][-1][0]+=1
        self.assertEqual(m.physical_hash(a),m.physical_hash(b))
        b['initial_state'][0][0]+=.001;self.assertNotEqual(m.physical_hash(a),m.physical_hash(b))

    def test_actual_opponent_settings_and_precapture_equality(self):
        for opponent in m.OPPONENTS:
            task=self.task('S',opponent=opponent,late=False);p,state,own,other=m.make_pair(task)
            expected=own.policy(p,state);rng=own.solver.get_rng_state(p)
            for uid in ('F','G'):
                q,s,a,b=m.make_pair(dict(task,uid=uid,effective=m.effective(uid,self.reg)))
                np.testing.assert_array_equal(a.policy(q,s),expected);self.assertEqual(rng,a.solver.get_rng_state(q))
            self.assertEqual(other.actual_settings['method'],opponent)
            if opponent!='Policy Only':self.assertEqual(other.actual_settings['depth'],100)

    def test_snapshot_remaining_time_worker_count_and_cache(self):
        tasks=[self.task(uid,map_id='asymmetric') for uid in ('S','G')]
        serial=[m.evaluate_one(t) for t in tasks]
        with mp.get_context('spawn').Pool(2) as pool:parallel=pool.map(m.evaluate_one,tasks)
        for a,b in zip(serial,parallel):
            self.assertEqual(a['trajectory_sha256'],b['trajectory_sha256']);self.assertLessEqual(a['own_mcts_calls'],3)
            self.assertEqual(m.old.previous.science(a),m.old.previous.science(b))
        task=tasks[1];record=serial[1]['_record'];snap=record['initial'];p,_,own,other=m.make_pair(task)
        own.solver.set_rng_state(p,snap['own_rng']);other.solver.set_rng_state(p,snap['opponent_rng']);m.o.restore_rng(snap['python_rng'])
        row,d,_,_=m.old.simulate(p,snap['state'].copy(),own,other,snap['stats'])
        self.assertEqual(serial[1]['trajectory_sha256'],m.old.previous.trajectory_digest(d['trajectory']))
        with tempfile.TemporaryDirectory() as tmp:
            args=m.parse_args(['--smoke','--output',tmp]);item=dict(key=task['key'],uid='G',task=task)
            m.commit_row(args,serial[1]);saved,d=m.read_episode(args,item)
            self.assertEqual(saved['trajectory_sha256'],parallel[1]['trajectory_sha256'])
            Path(saved['episode_record']['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):m.read_cached(args,item)

    def rows(self):
        rows=[]
        for map_id in m.MAPS:
            for i in range(3):
                scene=map_id+str(i)
                for opponent in m.OPPONENTS:
                    for u in m.UIDS:
                        success=int(i<(1 if u in ('B','S') else 2))
                        rows.append(dict(uid=u,episode_hash=scene,opponent=opponent,search_seeds=[i,i+1],
                            layout=m.GROUPS[map_id],map_id=map_id,legal_double_capture=success,
                            pursuer_illegal=0,evader_illegal=0,capture_count=2 if success else 1,one_capture_timeout=1-success,
                            completion_cost=40 if success else 80,first_capture_time=20.,own_mcts_calls=40,
                            own_cost=dict(step_calls=1000,policy_evaluations=100,value_evaluations=0)))
        return rows

    def test_stratified_statistics_preserve_opponents_and_pairing(self):
        rows=self.rows();a=m.analyze(rows);b=m.analyze(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(a['physical_scenes'],12)
        self.assertAlmostEqual(a['sections']['four']['mean']['comparisons']['F-S']['delta'],1/3)
        self.assertTrue(a['decision']['G_noninferior_to_F'])
        self.assertFalse(a['decision']['G_lower_call_counts'])
        with self.assertRaises(ValueError):m.analyze(rows[:-1])
        with self.assertRaises(ValueError):m.analyze(rows+[rows[0]])
        rows[0]['search_seeds']=[999,999]
        with self.assertRaises(ValueError):m.analyze(rows)

if __name__=='__main__':unittest.main()
