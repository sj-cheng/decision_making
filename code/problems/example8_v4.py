"""Example8 with the memory_observation_v4 evader reward.

Evaders earn a fixed reward for every step each teammate stays free, and a
capture costs the same at any time.  The C++ search in cpp/build_v4 applies
the same change; keep EVADER_SURVIVAL_REWARD in sync with
build_v4_training_search.py.
"""
import numpy as np

from problems.example8 import Example8

EVADER_SURVIVAL_REWARD = 0.005


class Example8V4(Example8):

	def __init__(self):
		super().__init__()
		# Distance-to-nearest-pursuer progress pulls evaders into corners.
		self.evader_progress_reward_weight = 0.0
		self.evader_survival_reward = EVADER_SURVIVAL_REWARD

	def transition_reward(self, s, s_next):
		reward = super().transition_reward(s, s_next)
		evader_failed = any(
			not self.is_robot_inside_and_clear(s_next, robot) for robot in self.evaders
		)
		if evader_failed:
			return reward
		newly_captured = self.active_evader_count(s) - self.active_evader_count(s_next)
		if newly_captured > 0:
			t = min(s_next[self.time_idx, 0], self.tf)
			# The base class discounts the capture penalty by (1 - t/tf).
			reward[self.evaders, 0] -= 0.5 * newly_captured * t / self.tf
		reward[self.evaders, 0] += self.evader_survival_reward * self.active_evader_count(s_next)
		return reward

	def is_robot_inside_and_clear(self, state, robot):
		position_idxs = self.state_idxs[robot]
		inside = np.all(
			(state[position_idxs, 0] >= self.state_lims[position_idxs, 0])
			& (state[position_idxs, 0] <= self.state_lims[position_idxs, 1])
		)
		return bool(inside) and not self.check_obstacle_collision(state, robot)
