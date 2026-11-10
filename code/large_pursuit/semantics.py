"""One return convention for PPO, teachers and every search node."""
GAMMA = 1.0
STEP_COST = .001
SCHEMA = 'large-pursuit-v2'


def team_reward(cfg, before, after):
    """Pursuer reward; evader reward is its negative. Terminal continuation is zero."""
    return float(before.alive.sum()-after.alive.sum())/cfg.n-STEP_COST


def backed_return(rewards, continuation=0., gamma=GAMMA):
    value=float(continuation)
    for reward in reversed(rewards):value=float(reward)+gamma*value
    return value

# Reuse the existing JOINT/GEOM common potential coefficient (e45b_joint_nte.ETA).
# Normalize by original team size alongside the new capture-fraction objective.
ETA = .05


def pursuit_potential(public,state,belief):
    if state.t>=public.cfg.horizon or not state.alive.any():return 0.
    samples=belief.samples[state.alive]
    distances=public.coverage(state.p[None],samples.reshape(-1,2)).reshape(int(state.alive.sum()),belief.count)
    return -float(distances.mean(1).sum())/public.cfg.n


def shaped_reward(raw,previous_potential,next_potential):
    """Existing potential difference; telescopes to a policy-independent constant."""
    return raw+ETA*(GAMMA*next_potential-previous_potential)
