"""Deterministic public route descriptors for E44; no target truth input.

Region representatives reproduce frozen_d_search_interface's nearest weighted
centroid sample rule, with coordinate ties for permutation invariance.  Paths
use the SAME route/move functions. D's probabilities and export are untouched.
"""
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import safe
from l45_corner_shared_20261025.local_pursuit import route, move

MAX_ROUTES = 12
REGION_FLOOR = .01
DIRECT_SECTORS = 8
ROUTE_DIM = 8


@njit(cache=True)
def route_arrays(at, samples, weights, labels, speed, vertices, dist, boxes, motion_boxes):
    # Columns: normalized action(2), waypoint offset(2), region mass,
    # weighted route cost / 80, stop flag, direct flag.
    descriptors = np.zeros((MAX_ROUTES, ROUTE_DIM))
    mask = np.zeros(MAX_ROUTES)
    anchors = np.zeros((MAX_ROUTES, 2))
    keys = np.full(MAX_ROUTES, -999, np.int64)
    groups = np.full(64, -999, np.int64)
    masses = np.zeros(64)
    costs = np.zeros(64)
    actions = np.zeros((64, 2))
    waypoints = np.zeros((64, 2))
    representative_mass = np.zeros(64)
    number = 0
    count = np.zeros(2, np.int64)
    omitted = 0.
    for r in range(int(labels.max())+1):
        mass = 0.
        center = np.zeros(2)
        for i in range(64):
            if labels[i] == r:
                mass += weights[i]
                center += weights[i]*samples[i]
        if mass < REGION_FLOOR:
            omitted += mass
            continue
        center /= mass
        winner, distance = -1, 1e100
        for i in range(64):
            if labels[i] != r:
                continue
            d = np.sum((samples[i]-center)**2)
            tie = winner >= 0 and abs(d-distance) < 1e-14
            lexical = winner < 0 or samples[i, 0] < samples[winner, 0] or (samples[i, 0] == samples[winner, 0] and samples[i, 1] < samples[winner, 1])
            if d < distance-1e-14 or (tie and lexical):
                winner, distance = i, d
        cost, first = route(at, samples[winner], speed, vertices, dist, boxes, count)
        if cost >= 1000000:
            omitted += mass
            continue
        waypoint = vertices[first].copy() if first >= 0 else samples[winner].copy()
        delta = waypoint-at
        sector = int(np.floor((np.arctan2(delta[1], delta[0])+np.pi)/(2*np.pi/DIRECT_SECTORS))) % DIRECT_SECTORS
        key = 100+first if first >= 0 else sector
        action = move(at, samples[winner], speed, vertices, dist, boxes, count)
        if not safe(at, action, speed, motion_boxes, count):
            omitted += mass
            continue
        found = -1
        for g in range(number):
            if groups[g] == key:
                found = g
                break
        if found < 0:
            found = number
            groups[found] = key
            number += 1
        masses[found] += mass
        costs[found] += mass*cost
        # A modal representative, never the mean of competing route actions.
        if mass > representative_mass[found]:
            representative_mass[found] = mass
            actions[found] = action
            waypoints[found] = waypoint
    # Always retain stop; select up to 11 route groups by mass, key as tie.
    descriptors[0, 6] = 1.
    mask[0], keys[0] = 1., -100
    chosen = np.zeros(number, np.bool_)
    retained = 0.
    for slot in range(1, min(MAX_ROUTES, number+1)):
        best = -1
        for g in range(number):
            if chosen[g]:
                continue
            if best < 0 or masses[g] > masses[best] or (masses[g] == masses[best] and groups[g] < groups[best]):
                best = g
        chosen[best] = True
        retained += masses[best]
        keys[slot] = groups[best]
        mask[slot] = 1.
        anchors[slot] = actions[best]/speed
        descriptors[slot, :2] = anchors[slot]
        descriptors[slot, 2:4] = (waypoints[best]-at)/20.
        descriptors[slot, 4] = masses[best]
        descriptors[slot, 5] = costs[best]/masses[best]/80.
        descriptors[slot, 7] = 1. if groups[best] < 100 else 0.
    for g in range(number):
        if not chosen[g]:
            omitted += masses[g]
    return descriptors, mask, anchors, keys, omitted, int(count[0])


def hand_probabilities(features):
    p = features['routes'][:, 4].copy()*features['route_mask']
    if p.sum() > 0:
        p *= .95/p.sum()
        p[0] = .05
    else:
        p[0] = 1.
    return p
