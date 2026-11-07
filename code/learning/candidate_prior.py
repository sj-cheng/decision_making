"""Public-only, permutation-equivariant candidate scorer for the frozen-D pilot.

No teacher scores, selected teacher action, world identity or hidden state are
features. PyTorch is used offline; the exported MLP runs in NumPy online.
"""
from pathlib import Path
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import route, sensor_visible

SEEDS = (17, 29, 43)
EPOCHS = 80
TEMPERATURE = 1.
PRIOR_FLOOR = .1


@njit(cache=True)
def geometry_scores(at, actions, samples, weights, speed, vertices, dist, boxes):
    """Same score, order and ties as the immutable strong G."""
    result = np.zeros(len(actions))
    count = np.zeros(2, np.int64)
    for j in range(len(actions)):
        nxt = at+actions[j]
        for i in range(len(samples)):
            steps, _ = route(nxt, samples[i], speed, vertices, dist, boxes, count)
            result[j] += weights[i]*(-steps+int(sensor_visible(nxt, samples[i], boxes)))
    return result


@njit(cache=True)
def features(at, actions, samples, weights, boxes, speed, remaining,
             last_positive, positive_age, previous_motion):
    """64 public features per candidate, with explicit padded map masks.

    Frozen-D weighted samples are compressed to endpoint-relative moments,
    sensor coverage and a 5x5 histogram. No access to the generative world.
    """
    if len(boxes) > 2:
        raise ValueError('Pilot feature schema supports at most two rectangles')
    answer = np.zeros((len(actions), 64))
    for j in range(len(actions)):
        nxt = at+actions[j]
        x = answer[j]
        x[0:2] = at/10.
        x[2:4] = actions[j]/speed
        x[4] = remaining/80.
        x[5] = speed/1.3
        x[6:8] = (last_positive-at)/20.
        x[8] = positive_age/80.
        x[9:11] = previous_motion/speed
        for b in range(len(boxes)):
            x[11+5*b] = 1.
            x[12+5*b] = (boxes[b, 0, 0]-nxt[0])/20.
            x[13+5*b] = (boxes[b, 0, 1]-nxt[0])/20.
            x[14+5*b] = (boxes[b, 1, 0]-nxt[1])/20.
            x[15+5*b] = (boxes[b, 1, 1]-nxt[1])/20.
        mean = np.zeros(2)
        second = np.zeros(2)
        cross = 0.
        for i in range(len(samples)):
            delta = samples[i]-nxt
            w = weights[i]
            mean += w*delta
            second += w*delta**2
            cross += w*delta[0]*delta[1]
            distance = np.sqrt(np.sum(delta**2))
            x[26] += w*distance/20.
            x[27] += w*np.max(np.abs(delta))/20.
            x[28] += w*np.sum(np.abs(delta))/40.
            x[29] += w*int(sensor_visible(nxt, samples[i], boxes))
            x[30] += w*int(distance <= 6.)
            x[31] += w*int(distance < 1.)
            ix = min(4, max(0, int((delta[0]+20.)/8.)))
            iy = min(4, max(0, int((delta[1]+20.)/8.)))
            x[32+5*ix+iy] += w
            if w > 0:
                x[57] -= w*np.log(w)/np.log(64.)
            x[58] += w*w
        x[21:23] = mean/20.
        x[23:25] = np.sqrt(np.maximum(0., second-mean**2))/20.
        x[25] = (cross-mean[0]*mean[1])/400.
        x[59:61] = nxt/10.
        x[61] = np.sqrt(np.sum(actions[j]**2))/(speed*np.sqrt(2.))
        x[62] = (10.-np.max(np.abs(nxt)))/10.
        x[63] = 1.
    return answer


def softmax(values):
    values = np.asarray(values, float)
    p = np.exp(values-values.max())
    return p/p.sum()


def mixed_prior(logits):
    return (1.-PRIOR_FLOOR)*softmax(logits)+PRIOR_FLOOR/len(logits)


class Network:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {k: data[k].copy() for k in data.files}

    def __call__(self, x):
        a = self.arrays
        z = np.clip((x-a['mean'])/a['scale'], -10., 10.)
        z = np.maximum(0., z@a['w0'].T+a['b0'])
        z = np.maximum(0., z@a['w1'].T+a['b1'])
        return (z@a['w2'].T+a['b2']).reshape(-1)
