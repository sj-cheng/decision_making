"""Paper figures for online deployment.

The action field and value field reuse example8.pretty_plot's scene: one robot is
moved over the map, and the memory snapshot is taken from a real rollout at the
moment that robot actually saw an opponent. The two networks are drawn on
separate figures, and the policy arrows are colored by action magnitude.

The tree is one real pursuer-turn expansion of that same state. Edges are
actions drawn from the policy prior; leaves are estimated by the value network.
"""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from learning.oracles import get_oracles
from problems.example8 import Example8

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../current/plots/paper"))
MODEL_DIR = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "../current/memory_observation_v3/models")
)
ITERATION = 45
QUERIED_ROBOT = 2


def load_models(problem):
	policy_paths = [
		os.path.join(MODEL_DIR, "model_policy_l{}_i{}.pt".format(ITERATION, robot))
		for robot in range(problem.num_robots)
	]
	value_path = os.path.join(MODEL_DIR, "model_value_l{}.pt".format(ITERATION))
	return get_oracles(
		problem,
		value_oracle_name="deterministic",
		value_oracle_path=value_path,
		policy_oracle_name="gaussian",
		policy_oracle_paths=policy_paths,
	)


def rollout(problem, policies, seed):
	np.random.seed(seed)
	state = problem.initialize()
	states = [state]
	for _ in range(40):
		action = np.zeros((problem.action_dim, 1))
		for robot in range(problem.num_robots):
			if not problem.is_active(state, robot):
				continue
			action[problem.action_idxs[robot], :] = policies[robot].eval(
				problem, state, robot, stochastic=False
			)
		state = problem.step(state, action, problem.dt)
		states.append(state)
		if problem.is_terminal(state):
			break
	return states


def rollout_with_sighting(problem, policies):
	best = None
	best_score = -1.0
	for seed in range(12):
		states = rollout(problem, policies, seed)
		reference, score = problem.memory_reference_state(states, QUERIED_ROBOT)
		if score > best_score:
			best = (states, reference, score, seed)
			best_score = score
		if score > 0.5:
			break
	return best


def mean_actions(policy, problem, states, robot):
	encodings = np.stack([problem.policy_encoding(state, robot).reshape(-1) for state in states])
	with torch.no_grad():
		mu, _ = policy(torch.tensor(encodings, dtype=torch.float32), training=True)
		return policy.scale_action(mu).detach().numpy()


def value_column(value_net, problem, states, robot):
	encodings = np.stack([problem.value_encoding(state).reshape(-1) for state in states])
	with torch.no_grad():
		values = value_net(torch.tensor(encodings, dtype=torch.float32)).detach().numpy()
	return values[:, robot]


def save(fig, name):
	os.makedirs(OUT_DIR, exist_ok=True)
	fig.savefig(os.path.join(OUT_DIR, name + ".pdf"), bbox_inches="tight")
	fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=200, bbox_inches="tight")
	plt.close(fig)


def expand(policies, problem, value_net, state, rng):
	action = np.zeros((problem.action_dim, 1))
	for robot in problem.pursuers:
		torch.manual_seed(int(rng.integers(1, 10**6)))
		np.random.seed(int(rng.integers(1, 10**6)))
		action[problem.action_idxs[robot], :] = policies[robot].eval(
			problem, state, robot, stochastic=True
		)
	nxt = problem.step(state, action, problem.dt)
	if problem.is_terminal(nxt):
		value = np.zeros(problem.num_robots)
	else:
		value = value_net.eval(problem, nxt).reshape(-1)
	return {
		"state": nxt,
		"value": value,
		"pursuer_action": action[problem.action_idxs[QUERIED_ROBOT], 0].copy(),
	}


def plot_tree_schematic():
	"""Illustrative tree: policy prior grows the edges, value scores only the leaves."""
	norm = Normalize(vmin=0.15, vmax=0.90)
	cmap = plt.get_cmap("viridis")
	fig, ax = plt.subplots(figsize=(7.8, 5.2))
	ax.set_xlim(-0.04, 1.04)
	ax.set_ylim(-0.02, 1.02)
	ax.set_aspect("equal")
	ax.axis("off")
	ax.set_title("Guided tree expansion", pad=10, fontsize=13)
	ax.text(
		0.50, 0.955,
		r"new edges are actions sampled from $\pi_\theta$; only unexpanded leaves carry $V_\eta$",
		ha="center", va="center", fontsize=9, color="#333333",
	)

	root = np.array((0.50, 0.80))
	left = np.array((0.16, 0.48))
	center = np.array((0.50, 0.48))
	right = np.array((0.84, 0.48))
	glow = np.array((0.34, 0.14))
	ghigh = np.array((0.66, 0.14))
	nodes = {
		"s0": root,
		"left": left,
		"s1": center,
		"right": right,
		"glow": glow,
		"ghigh": ghigh,
	}
	# prior, action heading in degrees, label offset from the edge midpoint
	edges = (
		("s0", "left", 0.15, 210.0, (-0.13, 0.07)),
		("s0", "s1", 0.55, 80.0, (0.12, -0.01)),
		("s0", "right", 0.30, -15.0, (0.13, 0.07)),
		("s1", "glow", 0.28, 230.0, (-0.14, -0.03)),
		("s1", "ghigh", 0.72, 25.0, (0.14, -0.03)),
	)

	def leaf_ink(value):
		color = cmap(norm(value))
		luminance = 0.2 * color[0] + 0.7 * color[1] + 0.1 * color[2]
		return color, "black" if luminance > 0.55 else "white"

	def draw_edge(start, end, prior, heading, offset):
		width = 1.1 + 6.5 * prior
		ax.add_patch(FancyArrowPatch(
			start, end,
			arrowstyle="-|>",
			mutation_scale=11,
			linewidth=width,
			color="#444444",
			shrinkA=16,
			shrinkB=16,
			zorder=2,
		))
		start = np.asarray(start, dtype=float)
		end = np.asarray(end, dtype=float)
		on_edge = 0.50 * (start + end)
		offset = np.asarray(offset, dtype=float)
		anchor = on_edge + 0.45 * offset
		radians = np.deg2rad(heading)
		glyph = np.array((np.cos(radians), np.sin(radians)))
		ax.annotate(
			"",
			xy=anchor + 0.020 * glyph,
			xytext=anchor - 0.020 * glyph,
			arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=1.6, mutation_scale=8),
			zorder=4,
		)
		ax.text(
			on_edge[0] + offset[0],
			on_edge[1] + offset[1],
			r"$\pi_\theta\ {:.2f}$".format(prior),
			ha="center", va="center", fontsize=8, color="#9a3412", zorder=5,
			bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"),
		)

	def draw_node(center, text, face, text_color="black"):
		ax.add_patch(Circle(
			center, 0.046, facecolor=face, edgecolor="black", linewidth=1.05, zorder=3,
		))
		ax.text(center[0], center[1], text, ha="center", va="center", fontsize=9, color=text_color, zorder=4)

	for src, dst, prior, heading, offset in edges:
		draw_edge(nodes[src], nodes[dst], prior, heading, offset)
	draw_node(root, r"$s_0$", "white")
	draw_node(center, r"$s_1$", "white")
	for name, value in (("left", 0.22), ("right", 0.48), ("glow", 0.31), ("ghigh", 0.84)):
		face, ink = leaf_ink(value)
		draw_node(nodes[name], "{:.2f}".format(value), face, ink)

	mappable = ScalarMappable(norm=norm, cmap=cmap)
	colorbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.015)
	colorbar.set_label(r"leaf value $V_\eta(s)$")
	ax.legend(handles=[
		Line2D([0], [0], color="#444444", lw=4.6, label=r"thicker edge: larger $\pi_\theta(a|s)$"),
		Line2D([0], [0], color="#c2410c", lw=1.6, label=r"action sampled from $\pi_\theta$"),
		Line2D([0], [0], marker="o", color="none", markerfacecolor="#2a9d8f", markeredgecolor="black", markersize=9, label=r"leaf scored by $V_\eta$"),
	], loc="lower center", bbox_to_anchor=(0.48, -0.08), ncol=3, frameon=False, fontsize=8)
	fig.tight_layout()
	save(fig, "guided_tree_expansion_schematic")


def plot_tree(problem, policies, value_net, state, value_limits):
	rng = np.random.default_rng(7)
	children = [expand(policies, problem, value_net, state, rng) for _ in range(3)]
	expanded = int(np.argmax([child["value"][QUERIED_ROBOT] for child in children]))
	grandchildren = [
		expand(policies, problem, value_net, children[expanded]["state"], rng)
		for _ in range(2)
	]
	norm = Normalize(vmin=value_limits[0], vmax=value_limits[1])
	cmap = plt.get_cmap("viridis")

	fig, ax = plt.subplots(figsize=(7.6, 4.8))
	ax.set_xlim(-0.02, 1.02)
	ax.set_ylim(0.02, 1.0)
	ax.axis("off")
	ax.set_title("Guided tree expansion", pad=8)

	root = np.array((0.50, 0.84))
	child_pos = [np.array(p) for p in ((0.18, 0.50), (0.50, 0.50), (0.82, 0.50))]
	grand_pos = [
		np.array(p) for p in (
			((0.06, 0.16), (0.30, 0.16)),
			((0.38, 0.16), (0.62, 0.16)),
			((0.70, 0.16), (0.92, 0.16)),
		)[expanded]
	]

	def draw_edge(start, end, action):
		ax.add_patch(FancyArrowPatch(
			start, end,
			arrowstyle="-|>",
			mutation_scale=12,
			linewidth=1.3,
			color="#444444",
			shrinkA=18,
			shrinkB=18,
			zorder=2,
		))
		direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
		direction = direction / max(np.linalg.norm(direction), 1e-8)
		perp = np.array((-direction[1], direction[0]))
		mid = 0.5 * (np.asarray(start) + np.asarray(end)) + 0.075 * perp
		heading = np.asarray(action, dtype=float)
		heading = heading / max(np.linalg.norm(heading), 1e-8)
		glyph = 0.028
		ax.annotate(
			"",
			xy=mid + glyph * heading,
			xytext=mid - glyph * heading,
			arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=1.5),
			zorder=4,
		)
		ax.text(
			mid[0] + 0.045 * perp[0],
			mid[1] + 0.045 * perp[1],
			r"$\pi_\theta$",
			ha="center",
			va="center",
			fontsize=10,
			color="#9a3412",
			zorder=4,
		)

	def draw_node(center, text, face, text_color="black"):
		ax.add_patch(Circle(center, 0.052, facecolor=face, edgecolor="black", linewidth=1.1, zorder=3))
		ax.text(center[0], center[1], text, ha="center", va="center", fontsize=8, color=text_color, zorder=4)

	def leaf_face(value):
		color = cmap(norm(value))
		luminance = 0.2 * color[0] + 0.7 * color[1] + 0.1 * color[2]
		return color, "black" if luminance > 0.55 else "white"

	draw_node(root, r"$s_0$", "white")
	for index, (child, position) in enumerate(zip(children, child_pos)):
		draw_edge(root, position, child["pursuer_action"])
		if index == expanded:
			draw_node(position, r"$s_{}$".format(index + 1), "white")
			for grand, gpos in zip(grandchildren, grand_pos):
				draw_edge(position, gpos, grand["pursuer_action"])
				face, ink = leaf_face(grand["value"][QUERIED_ROBOT])
				draw_node(gpos, "{:.2f}".format(grand["value"][QUERIED_ROBOT]), face, ink)
		else:
			face, ink = leaf_face(child["value"][QUERIED_ROBOT])
			draw_node(position, "{:.2f}".format(child["value"][QUERIED_ROBOT]), face, ink)

	mappable = ScalarMappable(norm=norm, cmap=cmap)
	colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
	colorbar.set_label(r"leaf value $V_\eta(s)$, same scale as the value field")
	ax.legend(handles=[
		Line2D([0], [0], color="#444444", lw=1.3, label="tree edge"),
		Line2D([0], [0], color="#c2410c", lw=1.5, label=r"action drawn from $\pi_\theta$"),
		Line2D([0], [0], marker="o", color="none", markerfacecolor="#2a9d8f", markeredgecolor="black", markersize=9, label=r"leaf estimated by $V_\eta$"),
	], loc="lower center", bbox_to_anchor=(0.46, -0.06), ncol=3, frameon=False, fontsize=9)
	fig.tight_layout()
	save(fig, "guided_tree_expansion")


def main():
	plt.rcParams.update({
		"font.size": 11,
		"axes.titlesize": 12,
		"axes.labelsize": 11,
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
	})
	problem = Example8()
	problem.use_minco_dynamics = True
	policies, value_net = load_models(problem)
	states, reference, saw_target, seed = rollout_with_sighting(problem, policies)
	points, field_states = problem.sample_robot_field(reference, QUERIED_ROBOT, resolution=36)
	coarse = np.arange(0, len(points), 3)
	actions = mean_actions(policies[QUERIED_ROBOT], problem, [field_states[i] for i in coarse], QUERIED_ROBOT)
	values = value_column(value_net, problem, field_states, QUERIED_ROBOT)

	save(problem.plot_policy_field(points[coarse], actions, reference, QUERIED_ROBOT), "policy_action_field")
	save(problem.plot_value_field(points, values, reference, QUERIED_ROBOT), "value_field")
	plot_tree(problem, policies, value_net, reference, (float(np.min(values)), float(np.max(values))))
	print("seed", seed, "steps", len(states), "saw target", saw_target)
	print("field points", len(points), "value range", float(np.min(values)), float(np.max(values)))
	print("wrote", OUT_DIR)


if __name__ == "__main__":
	main()
