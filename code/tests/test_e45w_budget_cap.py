import copy
import pytest
from finish_e45w_budget_cap import finalization_budget,CAP,BLOCK


def source():
    return dict(status='failed',counts=dict(team_transitions=CAP,rollouts_collected=976)),dict(
        traceback='terminal = audit.evaluate(model.num_timesteps)')


def test_no_more_sampling_is_added_to_a_certified_validation_failure():
    status,failure=source();before=copy.deepcopy(status)
    b=finalization_budget(status,1843200,failure)
    assert b['charged_lost_steps']==155648
    assert b['maximum_lifetime_team_steps']==CAP
    assert b['target_logical_steps']==1843200
    assert b['new_training_team_steps']==0 and b['new_optimizer_steps']==0
    assert not b['true_cap_actor_available']
    assert status==before


def test_a_sampling_failure_cannot_be_relabeled_as_final_validation():
    status,failure=source();failure['traceback']='sampling worker failed'
    with pytest.raises(RuntimeError,match='terminal validation'):finalization_budget(status,1843200,failure)


def test_an_unspent_or_nonintegral_budget_cannot_be_finalized():
    status,failure=source();status['counts']['team_transitions']-=BLOCK
    with pytest.raises(RuntimeError,match='fully spent'):finalization_budget(status,1843200,failure)
    status,failure=source()
    with pytest.raises(RuntimeError,match='checkpoint'):finalization_budget(status,CAP+BLOCK,failure)
