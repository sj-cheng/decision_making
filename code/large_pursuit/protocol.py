"""The user's fixed method families; aliases are checkpoint-matched direct controls."""
from .methods import METHODS,METHOD_BRANCH,DIRECT_METHODS
from .semantics import GAMMA,STEP_COST,ETA,SCHEMA

# These aliases add no trained branch and have no extra weight in self-play.
DIRECT_ALIASES={'NET_PRIOR':'NET with PRIOR checkpoint','NET_FUSION':'NET with FUSION checkpoint'}
POOL_METHODS=tuple(m for m in METHODS if m not in DIRECT_ALIASES)
PRIMARY=(('FUSION','GEOM'),('FUSION','AG'),('FUSION','PRIOR'),('POOL_LEARNED','POOL_GEOM'),
         ('JOINT','NET'),('PRIOR','NET_PRIOR'),('FUSION','NET_FUSION'),
         ('POOL_LEARNED','RANK_DIRECT'),('POOL_GEOM','GEOM_DIRECT'))


def method_protocol():
    return dict(schema=SCHEMA,methods=METHODS,pool_methods=POOL_METHODS,direct_aliases=DIRECT_ALIASES,
        branches=['prior','joint','fusion','rank'],checkpoint_family=METHOD_BRANCH,
        IG='independent minimum expected public route cost; per-target G action',
        AG='rectangular bottleneck matching, minimum sum tie break; unmatched pursuers support nearest target; per-target G action',
        random_MCTS='global legal proposals at every node; no pursuit geometry in pursuer proposals',
        GEOM='existing .75 independent belief/route/corner proposals + .25 global; common progressive UCT',
        neural='PRIOR/JOINT use .75 learned proposals + .25 global; JOINT_NV shares JOINT checkpoint with leaf value disabled',
        fusion='each bounded batch: 8 global, 12 learned, 12 existing GEOM-waypoint proposals; common progressive UCT, no additional value/ranking component',
        same_pool='common 32-array: 8 global + 24 existing geometry; uniform/geometric/learned expansion ordering; corresponding highest-score direct control',
        common_model='fixed reference: anchored .5m legal-edge D; common filter/tree: its continuous Gaussian conditioned on lawful actions; same nine intents, persistence .9, sigma .35, shared routing and exact-positive public re-anchor; no mixture',
        search='public observation branches; complete expansion-to-leaf trajectory with maximum depth8; real/filter/tree use same predict/condition and physical capture',
        return_convention=dict(gamma=GAMMA,capture='newly captured / original team size',step_cost=STEP_COST,existing_potential_coefficient=ETA,potential='negative expected min-route sum / original team size; terminal potential0; same telescoping convention in teacher/train/tree',terminal_continuation=0),
        pursuer_training='search supervision initialization and alternating search policy iteration; critic fitted only to actual team returns',
        evader_training='shared-parameter team PPO; full positions, alive masks, map and task clock',
        validation='all validation maps x two initials; fixed balanced assignment of methods and seed/history snapshots; select worst-opponent mean then overall mean, earliest tie',
        independent_evaders='61/73/89; train only on training maps against frozen final pursuer pool; fixed final checkpoint, no selection feedback to pursuers',
        statistics=dict(primary=PRIMARY,clusters='map with both initials',simultaneous='all primary contrasts across scales, ID/OOD, seeds, opponents and budgets'),
        reference_RootPUCT='variable-size alternating team max/min; simultaneous physical transition after both commitments; legacy C_pw2 alpha_pw.5 C_exp1 alpha_exp.25, tree depth100 alternating plies plus up to100 default-rollout plies, 3000 complete simulations, original uniform tree and lawful random rollout samplers, no added chase/flee heuristic; legacy RNG stream is not claimed bit-identical')


def episode_count():
    return 2*112*2*3*5*(len(DIRECT_METHODS)+4*(len(METHODS)-len(DIRECT_METHODS)))
