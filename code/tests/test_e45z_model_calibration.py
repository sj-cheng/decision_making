import numpy as np
from run_e45z_model_calibration import (
    conditional_moments, conditional_error, event_vectors, energy_distance)
from learning.e45x_event_diagnostic import target_step


def transition():
    points=np.array([[-1.,0.],[1.,0.]])
    links=np.tile([0,1],(2,1)).astype(np.int32)
    lengths=np.full(2,2,np.int32)
    probabilities=np.zeros((2,2,9))
    probabilities[:,0,0]=1.;probabilities[:,1,1:]=1.
    joint=np.zeros((2,9));joint[0,0]=.25;joint[1,1]=.75
    return (points,links,lengths,probabilities),joint


def test_total_variance_matches_exact_mode_and_position_mixture():
    graph,joint=transition();result=conditional_moments(graph,joint)
    # Probability of the left successor includes the 10% mode refresh.
    left=np.array([.9+.1/9,.1/9])
    expected=1-2*left
    np.testing.assert_allclose(result['conditional_mean'][:,0],expected)
    mean=np.dot([.25,.75],expected)
    assert np.isclose(result['predictive_mean'][0],mean)
    assert np.isclose(result['total_variance'],1-mean**2)
    assert np.isclose(result['current_position_variance'],.75)


def test_conditional_moments_agree_with_actual_D_sampler():
    graph,joint=transition();result=conditional_moments(graph,joint)
    rng=np.array([20261104],np.int64)
    samples=[graph[0][target_step(0,0,graph[1],graph[2],graph[3],rng)[0],0] for _ in range(12000)]
    assert abs(np.mean(samples)-result['conditional_mean'][0,0])<.025


def test_mean_bias_correction_retains_negative_finite_sample_estimates():
    reference=np.array([[[-1.,0.],[1.,0.]]])
    result=conditional_error(np.zeros((1,2)),reference)
    assert result['conditional_mse']==1.
    assert result['conditional_mean_bias_squared_unbiased']==-1.
    assert result['reference_random_variance']==2.
    assert np.isclose(result['conditional_mse'],result['conditional_mean_bias_squared_unbiased']+result['reference_random_variance'])


def test_double_capture_events_include_retirement_and_remove_visibility():
    result=event_vectors(np.array([[0.,0.],[3.,0.]]),np.array([[.5,0.],[3.5,0.]]),
        np.zeros((1,2,2)),np.ones(2,bool),np.ones(2,bool),np.zeros((0,2,2)))
    np.testing.assert_array_equal(result,[[1,1,1,1,0,0,1,1]])


def test_energy_distance_zero_for_equal_samples_and_positive_for_shift():
    a=np.zeros((4,2));b=np.tile([3.,4.],(5,1))
    assert energy_distance(a,a,'cpu')==0.
    assert energy_distance(a,b,'cpu')==10.
