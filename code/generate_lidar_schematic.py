import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

from problems.problem import get_problem


CHANNELS = [
	("team", "same-team robot", "#2b6cb0"),
	("opponent", "opponent robot", "#d62728"),
	("obstacle", "obstacle", "#4a5568"),
	("boundary", "workspace boundary", "#718096"),
]


def make_demo_state(problem):
	state = np.zeros((problem.state_dim, 1), dtype=float)

	positions = [
		(-6.0, 0.0),   # ego evader
		(-5.0, 3.0),   # same-team evader
		(-1.0, 0.35),  # opponent mostly occluded by the left obstacle
		(-3.0, -3.6),  # visible opponent
	]
	for robot, pos in enumerate(positions):
		state[problem.state_idxs[robot], 0] = pos

	state[problem.time_idx, 0] = 12.0
	for idx in problem.active_idxs:
		state[idx, 0] = 1.0
	return state


def draw_obstacles(ax, problem, alpha=0.78):
	for obstacle in problem.obstacles:
		rect = Rectangle(
			(obstacle[0, 0], obstacle[1, 0]),
			obstacle[0, 1] - obstacle[0, 0],
			obstacle[1, 1] - obstacle[1, 0],
			facecolor="#6b7280",
			edgecolor="#374151",
			linewidth=0.8,
			alpha=alpha,
			zorder=5,
		)
		ax.add_patch(rect)


def channel_distances(problem, state, ego_robot):
	lidar = problem.semantic_lidar(state, ego_robot)
	distances = lidar.reshape(problem.lidar_num_beams, problem.lidar_num_channels)
	return distances * problem.lidar_max_range()


def nearest_channels(distances, max_range):
	detected = distances < max_range * (1.0 - 1e-8)
	nearest = np.argmin(distances, axis=1)
	has_hit = detected[np.arange(distances.shape[0]), nearest]
	return nearest, has_hit


def draw_robot(ax, position, color, label, marker="o", alpha=1.0, zorder=20):
	kwargs = {
		"s": 72,
		"marker": marker,
		"color": color,
		"alpha": alpha,
		"zorder": zorder,
		"label": label,
	}
	if marker != "x":
		kwargs.update({"edgecolor": "white", "linewidth": 0.9})
	ax.scatter(position[0], position[1], **kwargs)


def draw_world_panel(ax, problem, state, ego_robot, distances, display_range):
	origin = state[problem.state_idxs[ego_robot], 0]
	directions = problem.lidar_directions
	max_range = problem.lidar_max_range()
	nearest, has_hit = nearest_channels(distances, max_range)

	all_segments = [
		[origin, origin + display_range * direction]
		for direction in directions
	]
	ax.add_collection(
		LineCollection(all_segments, colors="#cbd5e1", linewidths=0.35, alpha=0.45, zorder=1)
	)
	ax.add_patch(
		Circle(origin, display_range, facecolor="#60a5fa", edgecolor="#2563eb", linewidth=0.9, alpha=0.08, zorder=0)
	)

	for channel, (_, _, color) in enumerate(CHANNELS):
		segments = []
		for beam_idx, direction in enumerate(directions):
			if nearest[beam_idx] != channel or not has_hit[beam_idx]:
				continue
			distance = min(distances[beam_idx, channel], display_range)
			if channel == 3 and distance >= display_range * 0.98:
				continue
			segments.append([origin, origin + distance * direction])
		if segments:
			ax.add_collection(
				LineCollection(
					segments,
					colors=color,
					linewidths=1.25 if channel != 3 else 0.7,
					alpha=0.86 if channel != 3 else 0.42,
					zorder=8,
				)
			)

	draw_obstacles(ax, problem)

	robot_colors = ["#1f77b4", "#2b6cb0", "#d62728", "#b91c1c"]
	robot_labels = ["ego robot", "same-team robot", "occluded opponent", "visible opponent"]
	robot_markers = ["*", "o", "x", "o"]
	robot_alpha = [1.0, 1.0, 0.42, 1.0]
	for robot in range(problem.num_robots):
		pos = state[problem.state_idxs[robot], 0]
		draw_robot(
			ax,
			pos,
			robot_colors[robot],
			robot_labels[robot],
			marker=robot_markers[robot],
			alpha=robot_alpha[robot],
			zorder=30 if robot == ego_robot else 22,
		)

	occluded_pos = state[problem.state_idxs[2], 0]
	ax.plot(
		[origin[0], occluded_pos[0]],
		[origin[1], occluded_pos[1]],
		color="#9ca3af",
		linestyle=(0, (3, 2)),
		linewidth=0.9,
		alpha=0.7,
		zorder=4,
	)
	ax.text(-2.15, 1.92, "obstacle blocks\nthis robot", fontsize=7.5, color="#374151", ha="center")

	ax.set_xlim(-8.2, 1.4)
	ax.set_ylim(-5.4, 4.7)
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel("global x")
	ax.set_ylabel("global y")
	ax.set_title("World frame ray casting", fontsize=10)
	ax.grid(True, color="#e5e7eb", linewidth=0.55)
	ax.legend(
		loc="lower left",
		frameon=True,
		framealpha=0.92,
		borderpad=0.35,
		handletextpad=0.45,
		fontsize=6.6,
	)


def draw_local_panel(ax, problem, distances, display_range):
	directions = problem.lidar_directions
	max_range = problem.lidar_max_range()
	nearest, has_hit = nearest_channels(distances, max_range)

	for radius in [2.0, 4.0, 6.0]:
		ax.add_patch(Circle((0, 0), radius, facecolor="none", edgecolor="#e5e7eb", linewidth=0.7, zorder=0))
		ax.text(radius / np.sqrt(2), radius / np.sqrt(2), "{:.0f}".format(radius), fontsize=7, color="#6b7280")

	spoke_segments = [
		[(0.0, 0.0), tuple(display_range * direction)]
		for idx, direction in enumerate(directions)
		if idx % 4 == 0
	]
	ax.add_collection(LineCollection(spoke_segments, colors="#d1d5db", linewidths=0.45, alpha=0.55, zorder=1))

	for channel, (_, label, color) in enumerate(CHANNELS):
		points = []
		for beam_idx, direction in enumerate(directions):
			if nearest[beam_idx] != channel or not has_hit[beam_idx]:
				continue
			distance = distances[beam_idx, channel]
			if distance > display_range:
				continue
			if channel == 3 and distance >= display_range * 0.98:
				continue
			points.append(distance * direction)
		if points:
			points = np.asarray(points)
			ax.scatter(points[:, 0], points[:, 1], s=18, color=color, alpha=0.9, linewidths=0, label=label, zorder=12)

	ax.scatter(0, 0, marker="*", s=130, color="#1f77b4", edgecolor="white", linewidth=0.9, zorder=20)
	ax.arrow(0, 0, 1.15, 0, width=0.025, head_width=0.22, head_length=0.24, color="#111827", zorder=21)
	ax.text(1.42, -0.14, "ego x", fontsize=7.5, color="#111827")
	ax.set_xlim(-display_range, display_range)
	ax.set_ylim(-display_range, display_range)
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel("local x")
	ax.set_ylabel("local y")
	ax.set_title("Ego-frame semantic returns", fontsize=10)
	ax.grid(False)


def draw_encoding_panel(ax, problem, distances):
	normalized = np.clip(distances / problem.lidar_max_range(), 0.0, 1.0).T
	image = ax.imshow(
		normalized,
		aspect="auto",
		interpolation="nearest",
		cmap="viridis_r",
		vmin=0.0,
		vmax=1.0,
		extent=(0, 360, problem.lidar_num_channels - 0.5, -0.5),
	)
	ax.set_yticks(np.arange(problem.lidar_num_channels))
	ax.set_yticklabels([label for _, label, _ in CHANNELS], fontsize=7.5)
	ax.set_xticks([0, 90, 180, 270, 360])
	ax.set_xlabel("beam angle (deg)")
	ax.set_title("Neural-network input: 4 x 64 normalized ranges", fontsize=10)
	ax.tick_params(axis="x", labelsize=7.5)
	return image


def main():
	problem = get_problem("example8")
	ego_robot = 0
	state = make_demo_state(problem)
	distances = channel_distances(problem, state, ego_robot)
	display_range = 6.4

	plt.rcParams.update({
		"font.size": 8.5,
		"axes.linewidth": 0.8,
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
	})

	fig = plt.figure(figsize=(7.35, 4.45), constrained_layout=True)
	gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.05, 0.72])
	ax_world = fig.add_subplot(gs[:, 0])
	ax_local = fig.add_subplot(gs[0, 1])
	ax_encoding = fig.add_subplot(gs[1, 1])

	draw_world_panel(ax_world, problem, state, ego_robot, distances, display_range)
	draw_local_panel(ax_local, problem, distances, display_range)
	image = draw_encoding_panel(ax_encoding, problem, distances)

	cbar = fig.colorbar(image, ax=ax_encoding, fraction=0.045, pad=0.02)
	cbar.set_label("normalized distance", fontsize=7.5)
	cbar.ax.tick_params(labelsize=7)

	out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "current", "plots"))
	os.makedirs(out_dir, exist_ok=True)
	pdf_path = os.path.join(out_dir, "lidar_perception_schematic.pdf")
	png_path = os.path.join(out_dir, "lidar_perception_schematic.png")
	fig.savefig(pdf_path, bbox_inches="tight")
	fig.savefig(png_path, dpi=240, bbox_inches="tight")
	plt.close(fig)
	print("saved {}".format(pdf_path))
	print("saved {}".format(png_path))


if __name__ == "__main__":
	main()
