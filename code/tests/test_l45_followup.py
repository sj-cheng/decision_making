"""Behavioral tests for the isolated l45 follow-up; outputs are temporary."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
import run_l45_ppo_followup as s


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary=tempfile.TemporaryDirectory(prefix='l45_followup_unit_')
        cls.args=s.parse_args(['--smoke','--output',cls.temporary.name,'--workers','1'])
        cls.banks=s.prepare(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def late_spec(self):
        spec=copy.deepcopy(self.banks['screen_four'][0])
        p,state=s.o.restore_scenario(spec)
        state[p.time_idx,0]=p.tf-p.dt
        spec['initial_state']=state.tolist()
        spec['episode_hash']=s.o.json_hash({k:v for k,v in spec.items() if k!='episode_hash'})
        return spec

    def test_cpp_betas_and_opponent_are_independent_and_fixed(self):
        models=s.fixed_models(self.args,'novalue')
        item=s.requests(self.args,[models[1]],[self.late_spec()],opponents=('Search + Learning',))[0]
        row=s.evaluate_one(item['task'])
        self.assertEqual(row['actual_own']['beta_value'],0.)
        self.assertAlmostEqual(row['actual_own']['beta_policy'],.5)
        self.assertAlmostEqual(row['actual_opponent']['beta_value'],.1)
        self.assertAlmostEqual(row['actual_opponent']['beta_policy'],.5)
        for cfg,(bp,bv) in s.CONFIGS.items():
            p,_=s.o.restore_scenario(self.late_spec())
            sol=s.make_solver(p,self.args.output/'models/baseline','Search + Learning',1,s.settings(self.args,cfg),3)
            self.assertAlmostEqual(sol.actual_settings['beta_policy'],bp)
            self.assertAlmostEqual(sol.actual_settings['beta_value'],bv)
        row=s.evaluate_one(dict(item['task'],opponent='Search Only'))
        self.assertEqual(row['actual_opponent']['budget'],3)
        self.assertEqual(row['actual_opponent']['beta_policy'],0.)
        self.assertEqual(row['actual_opponent']['beta_value'],0.)

    def test_original_configuration_and_zero_prior_negative_control(self):
        result=s.audit(self.args,self.banks)
        self.assertTrue(result['original_configuration_exact_match'])
        self.assertTrue(result['zero_policy_prior_exact_match'])

    def test_completion_bonus_only_once_legal_terminal_and_zero_control(self):
        original=dict(rows=[dict(reward=.1,done=False),dict(reward=.2,done=True)],
                      stats=dict(legal_double_capture=1,environment_return=.3))
        zero=s.shape_rewards(copy.deepcopy(original),0.)
        self.assertEqual([r['reward'] for r in zero['rows']],[.1,.2])
        positive=s.shape_rewards(copy.deepcopy(original),1.)
        self.assertEqual([r['reward'] for r in positive['rows']],[.1,1.2])
        self.assertAlmostEqual(positive['stats']['environment_return'],.3)
        self.assertAlmostEqual(positive['stats']['training_return'],1.3)
        twice=s.shape_rewards(positive,1.)
        self.assertEqual([r['reward'] for r in twice['rows']],[.1,1.2])
        for reason in ('one_capture_timeout','pursuer_illegal','evader_illegal'):
            failed=copy.deepcopy(original)
            failed['stats'].update(legal_double_capture=0,**{reason:1})
            shaped=s.shape_rewards(failed,1.)
            self.assertEqual([r['completion_bonus'] for r in shaped['rows']],[0.,0.])
        malformed=copy.deepcopy(original); malformed['rows'][-1]['done']=False
        with self.assertRaisesRegex(ValueError,'must terminate'):
            s.shape_rewards(malformed,1.)

    def test_real_zero_bonus_reproduces_original_sampling_and_probability(self):
        initial=s.o.load_torch(self.args.output/'initial/seed0.pt')
        path=self.args.output/'zero_bonus_snapshot.pt'
        s.o.save_torch(path,dict(actors=initial['actors'],episode=0))
        task=dict(spec=self.late_spec(),snapshot=str(path),snapshot_episode=0,
            baseline=str(self.args.output/'models/baseline'),opponent='Policy Only',
            settings=s.settings(self.args),sampling_seed=19,completion_bonus=0.)
        original=s.o.rollout(task); adapted=s.reward_rollout(task)
        self.assertEqual(original['stats']['environment_return'],adapted['stats']['environment_return'])
        for left,right in zip(original['rows'],adapted['rows']):
            for key in left:self.assert_nested(left[key],right[key])
            self.assertEqual(right['completion_bonus'],0.)

    def test_formal_search_matrix_and_locked_selection_cannot_be_changed(self):
        args=s.parse_args([])
        scenes=[dict(episode_hash=f'scene{i}',layout='four' if i<40 else 'none') for i in range(60)]
        models=s.fixed_models(self.args)
        items=[r for cfg in s.CONFIGS for r in s.requests(args,
               [dict(m,config=cfg) for m in models],scenes)]
        self.assertEqual(len(items),2880)
        self.assertEqual(len({r['key'] for r in items}),2880)
        for item in items:
            self.assertEqual(item['task']['own_settings']['search_budget'],1000)
            self.assertEqual(item['task']['opponent_settings']['search_only_budget'],3000)
            self.assertEqual(item['task']['opponent_settings']['beta_policy'],.5)
            self.assertEqual(item['task']['opponent_settings']['beta_value'],.1)
        with tempfile.TemporaryDirectory(prefix='l45_followup_selection_') as directory:
            path=Path(directory)/'selection.json'
            s.lock_json(path,dict(selected='base'))
            self.assertEqual(s.locked_read(path),dict(selected='base'))
            s.o.write_json(path,dict(selected='high'))
            with self.assertRaisesRegex(ValueError,'digest changed'):s.locked_read(path)

    def test_physical_splits_and_paired_shared_initialization(self):
        identities=[s.previous.physical_hash(case) for bank in self.banks.values() for case in bank]
        self.assertEqual(len(identities),len(set(identities)))
        self.assertFalse(s.history_scenes().intersection(identities))
        self.assertEqual(len(self.banks['train0']),8)
        for seed in s.SEEDS:
            initial=s.o.load_torch(self.args.output/f'initial/seed{seed}.pt')
            actorsA=s.o.load_actors(s.o.original_problem(),states=initial['actors'])
            actorsB=s.o.load_actors(s.o.original_problem(),states=initial['actors'])
            for left,right in zip(actorsA,actorsB):
                for key in left.state_dict():
                    self.assertTrue(torch.equal(left.state_dict()[key],right.state_dict()[key]))
            critics=[s.o.TrainingCritic(),s.o.TrainingCritic()]
            for critic in critics:critic.load_state_dict(initial['critic'])
            for key in critics[0].state_dict():
                self.assertTrue(torch.equal(critics[0].state_dict()[key],critics[1].state_dict()[key]))
            s.o.restore_rng(initial['rng']); first=torch.randn(5)
            s.o.restore_rng(initial['rng']); second=torch.randn(5)
            torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_actual_tensor_aliases_share_one_episode_and_resume_cache(self):
        baseline=s.model('baseline',self.args.output/'models/baseline')
        alias=dict(baseline,label='same_weights_final',seed=0)
        items=s.requests(self.args,[baseline,alias],[self.late_spec()],opponents=('Policy Only',))
        self.assertEqual(items[0]['key'],items[1]['key'])
        with mock.patch.object(s,'evaluate_one',wraps=s.evaluate_one) as call:
            rows=s.run_evaluation(self.args,'alias_test',items,None,lambda *a,**kw:None)
            self.assertEqual(call.call_count,1)
            again=s.run_evaluation(self.args,'alias_test',items,None,lambda *a,**kw:None)
            self.assertEqual(call.call_count,1)
        self.assertEqual({r['label'] for r in rows},{'baseline','same_weights_final'})
        self.assertEqual(rows,again)

    def synthetic_screen(self):
        rows=[]
        for cfg in s.CONFIGS:
            for label in ('baseline','old0','old1','old2'):
                for layout,n in (('four',40),('none',20)):
                    for scene in range(n):
                        for opponent in s.o.OPPONENTS:
                            cutoff=15 if layout=='none' else 20
                            if layout=='four' and label!='baseline':
                                cutoff=dict(base=20,low=25,high=24,novalue=23)[cfg]
                            success=int(scene<cutoff)
                            rows.append(dict(label=label,config=cfg,mode='search',role=1,layout=layout,
                                episode_hash=f'{layout}{scene}',opponent=opponent,search_seeds=[scene,scene+1],
                                legal_double_capture=success,pursuer_illegal=0,evader_illegal=0,one_capture_timeout=1-success,
                                pursuer_return=0.,pursuer_latency_sum_ms=1.,pursuer_latency_count=1,
                                completion_cost=40. if success else 80.))
        return rows

    def test_search_selection_checks_same_configuration_and_each_seed(self):
        rows=self.synthetic_screen()
        self.assertEqual(s.select_search(rows)['selected'],'low')
        # A search-parameter-only gain cannot be attributed to PPO.
        for r in rows:
            if r['config']=='low' and r['label']=='baseline' and r['layout']=='four':
                r['legal_double_capture']=int(int(r['episode_hash'][4:])<26)
        selected=s.select_search(rows)
        self.assertFalse(next(d for d in selected['decisions'] if d['config']=='low')['eligible'])
        self.assertEqual(selected['selected'],'high')
        # Regression for one seed must not disappear inside the three-seed mean.
        for r in rows:
            if r['config']=='high' and r['label']=='old2' and r['layout']=='none':
                r['legal_double_capture']=0
        self.assertEqual(s.select_search(rows)['selected'],'novalue')

    def test_pairing_rejects_duplicate_missing_rng_and_gate_respects_ci(self):
        rows=self.synthetic_screen()
        self.assertTrue(s.improvement(dict(delta=.06,per_seed_delta=[.04,.06,.08],
                            own_illegal_delta=0.,ci95=[.01,.1]),.05,True))
        self.assertFalse(s.improvement(dict(delta=.06,per_seed_delta=[.04,.06,.08],
                            own_illegal_delta=0.,ci95=[-.01,.1]),.05,True))
        self.assertFalse(s.regression(dict(per_seed_delta=[.1,.1,-.04],own_illegal_delta=0.)))
        refs=['baseline']*3; labels=['old0','old1','old2']
        victim=next(r for r in rows if r['label']=='old1' and r['config']=='low' and r['layout']=='four')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            s.compare(rows+[copy.deepcopy(victim)],labels,refs,cfg='low')
        with self.assertRaisesRegex(ValueError,'Unpaired'):
            s.compare([r for r in rows if r is not victim],labels,refs,cfg='low')
        victim['search_seeds']=[999,999]
        with self.assertRaisesRegex(ValueError,'Unpaired search'):
            s.compare(rows,labels,refs,cfg='low')

    def test_reward_checkpoint_score_tie_and_no_obstacle_exclusion(self):
        baseline=self.synthetic_screen()
        baseline=[r for r in baseline if r['label']=='baseline' and r['config']=='base']
        candidate=copy.deepcopy(baseline)
        self.assertEqual(s.checkpoint_score(candidate),s.checkpoint_score(baseline))
        for row in candidate:
            if row['layout']=='none':row['legal_double_capture']=0
        self.assertFalse(s.checkpoint_eligible(candidate,baseline))

    def test_fingerprints_source_protection_and_checkpoint_tampering(self):
        path=self.args.output/'scenarios.json'; original=path.read_bytes()
        try:
            changed=s.o.read_json(path); changed['train0'][0]['search_seeds'][0]+=1
            s.o.write_json(path,changed)
            with self.assertRaisesRegex(ValueError,'signature changed'):s.verify(self.args)
        finally:path.write_bytes(original)
        parent=s.SOURCE
        with self.assertRaises(SystemExit):s.parse_args(['--output',str(parent/'forbidden')])
        with tempfile.TemporaryDirectory(prefix='l45_followup_checkpoint_') as temp:
            directory=Path(temp)
            payload=dict(signature=s.o.json_hash(s.o.read_json(self.args.output/'signature.json')),
                         experiment_sha=s.o.sha(self.args.output/'experiment.json'),episode=0,pending_validation=False)
            s.commit_checkpoint(directory,payload,{})
            s.load_checkpoint(self.args,directory)
            pointer=s.o.read_json(directory/'latest.json')
            file=Path(pointer['path']); file.write_bytes(file.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):s.load_checkpoint(self.args,directory)

    def test_pipeline_stops_or_advances_only_at_declared_boundaries(self):
        fake=dict(pursuit_pass=True,regression_pass=False)
        with mock.patch.object(s,'search_screen',return_value=dict(selected='low')), \
             mock.patch.object(s,'search_confirm',return_value=(fake,[],[],'low')), \
             mock.patch.object(s,'reward_train') as train, \
             mock.patch.object(s,'finish',return_value='regression-stop') as finish:
            self.assertEqual(s.pipeline(self.args,{},None,None),'regression-stop')
            train.assert_not_called()
            self.assertEqual(finish.call_args.args[1],'search_no_obstacle_regression_failed')
        fake=dict(pursuit_pass=False,regression_pass=True)
        with mock.patch.object(s,'search_screen',return_value=dict(selected='base')), \
             mock.patch.object(s,'search_confirm',return_value=(fake,[],[],'base')), \
             mock.patch.object(s,'reward_train',return_value=dict(selected=None)) as train, \
             mock.patch.object(s,'finish',return_value='finished') as finish:
            self.assertEqual(s.pipeline(self.args,{},None,None),'finished')
            train.assert_called_once()
            self.assertEqual(finish.call_args.args[1],'no_reward_validation_candidate')
        fake=dict(pursuit_pass=True,regression_pass=True)
        with mock.patch.object(s,'search_screen',return_value=dict(selected='base')), \
             mock.patch.object(s,'search_confirm',return_value=(fake,[],[],'base')), \
             mock.patch.object(s,'reward_train') as train, \
             mock.patch.object(s,'guards_and_latency',return_value='guard') as guard:
            self.assertEqual(s.pipeline(self.args,{},None,None),'guard')
            train.assert_not_called(); guard.assert_called_once()

    def test_stale_dashboard_is_not_reported_running(self):
        args=copy.copy(self.args)
        s.o.write_json(args.output/'status.json',dict(phase='reward_training',pid=99999999,workers=80))
        with mock.patch.object(s,'live_pids',return_value=[]):
            text=s.dashboard(args)
        self.assertIn('未运行',text)
        self.assertNotIn('状态：运行中',text)

    def test_real_evader_regression_and_serial_path_with_identical_models(self):
        four=self.late_spec()
        none=copy.deepcopy(self.banks['guard_none'][0])
        p,state=s.o.restore_scenario(none); state[p.time_idx,0]=p.tf-p.dt
        none['initial_state']=state.tolist()
        none['episode_hash']=s.o.json_hash({k:v for k,v in none.items() if k!='episode_hash'})
        banks=dict(guard_four=[four],guard_none=[none],latency_four=[four])
        baseline=s.model('baseline',self.args.output/'models/baseline')
        models=[baseline]+[dict(baseline,label=f'old{seed}',seed=seed) for seed in s.SEEDS]
        with mock.patch.object(s,'finish',return_value='verified-path') as finish:
            result=s.guards_and_latency(self.args,banks,None,lambda *a,**kw:None,
                models,[f'old{seed}' for seed in s.SEEDS],'base','unit-test',{})
        self.assertEqual(result,'verified-path')
        self.assertTrue(finish.call_args.args[-1])
        guard=s.o.read_json(self.args.output/'evader_regression/result.json')
        self.assertTrue(guard['passed'])
        for row in s.o.read_json(self.args.output/'serial_latency/summary.json'):
            self.assertIn(row['mode'],('search','policy'))
            self.assertGreaterEqual(row['decision_ms'],0.)

    def test_nondefault_search_confirmation_keeps_original_anchor_and_opponent(self):
        four=self.late_spec(); none=copy.deepcopy(self.banks['screen_none'][0])
        p,state=s.o.restore_scenario(none); state[p.time_idx,0]=p.tf-p.dt
        none['initial_state']=state.tolist()
        none['episode_hash']=s.o.json_hash({k:v for k,v in none.items() if k!='episode_hash'})
        with mock.patch.object(s,'locked_read',return_value=dict(selected='low')):
            result,models,labels,cfg=s.search_confirm(self.args,
                dict(search_four=[four],search_none=[none]),None,lambda *a,**kw:None)
        self.assertEqual(cfg,'low')
        self.assertEqual(labels,['old0','old1','old2'])
        self.assertIn('same',result); self.assertIn('anchor',result)
        manifest=s.o.read_json(self.args.output/'search_confirmation/manifest.json')
        self.assertEqual(len(manifest),54)
        self.assertEqual({r['config'] for r in manifest if r['label']=='baseline'},{'base','low'})
        task=next(r for r in manifest if r['label']=='old0' and r['config']=='low')
        row=s.o.read_json(self.args.output/'cache'/(task['key']+'.json'))
        if row['mode']=='search':self.assertAlmostEqual(row['actual_own']['beta_policy'],.25)
        self.assertIn(row['actual_opponent']['method'],s.o.OPPONENTS)

    def assert_nested(self,left,right):
        if isinstance(left,torch.Tensor):self.assertTrue(torch.equal(left,right))
        elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
        elif isinstance(left,dict):
            self.assertEqual(left.keys(),right.keys())
            for key in left:self.assert_nested(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.assert_nested(a,b)
        else:self.assertEqual(left,right)

    def test_interrupted_training_resumes_all_seeds_bitwise(self):
        with tempfile.TemporaryDirectory(prefix='l45_followup_resume_') as temporary:
            root=Path(temporary)
            def run(name,workers,stop=0):
                cmd=[sys.executable,str(CODE/'run_l45_ppo_followup.py'),'--smoke','--stage','reward-train',
                     '--output',str(root/name),'--workers',str(workers)]
                if stop:cmd+=['--max-updates',str(stop)]
                with (root/(name+f'-{stop}.log')).open('w') as log:
                    subprocess.run(cmd,cwd=str(CODE.parent),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
            run('continuous',2)
            run('resumed',2,1)
            state=s.o.read_json(root/'resumed/status.json')
            self.assertEqual(state['phase'],'stopped_at_checkpoint')
            self.assertEqual(state['episode'],4)
            run('resumed',3)
            for arm in ('A','B'):
                for seed in s.SEEDS:
                    directories=[root/name/f'reward/{arm}/seed{seed}' for name in ('continuous','resumed')]
                    payload=[]
                    for directory in directories:
                        pointer=s.o.read_json(directory/'latest.json')
                        payload.append(s.o.load_torch(pointer['path']))
                    for field in ('actors','critic','actor_optimizer','critic_optimizer','rng','episode','best_episode','best_score','pending_validation'):
                        with self.subTest(arm=arm,seed=seed,field=field):
                            self.assert_nested(payload[0][field],payload[1][field])
                    baseline=s.o.model_fingerprint(root/'continuous/models/baseline')
                    final=s.o.model_fingerprint(directories[0]/'models_ep8')
                    for name in (*s.o.MODEL_NAMES[:2],s.o.MODEL_NAMES[-1]):
                        self.assertEqual(baseline[name],final[name])


if __name__=='__main__':unittest.main()
