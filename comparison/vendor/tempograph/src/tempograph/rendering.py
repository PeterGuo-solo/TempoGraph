"""Optional SMPL-style joint renderer for canonical motions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import MotionSequence


SMPL22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def world_joints(motion: MotionSequence) -> np.ndarray:
    if motion.local_joint_positions is None:
        raise ValueError("rendering requires local_joint_positions or an SMPL forward pass")
    return motion.local_joint_positions + motion.root_positions[:, None]


def render_motion(
    motion: MotionSequence,
    output: str | Path,
    parents: tuple[int, ...] = SMPL22_PARENTS,
    fps: float | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    except ModuleNotFoundError as error:
        raise RuntimeError("matplotlib is required for rendering") from error
    joints = world_joints(motion)
    if len(parents) != joints.shape[1]:
        raise ValueError("parent topology must match the joint count")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    minimum = joints.min(axis=(0, 1))
    maximum = joints.max(axis=(0, 1))
    center = (minimum + maximum) / 2
    radius = max(float(np.max(maximum - minimum)) * 0.6, 1.0)
    figure = plt.figure(figsize=(6, 6))
    axis = figure.add_subplot(111, projection="3d")
    lines = [axis.plot([], [], [], linewidth=2)[0] for parent in parents if parent >= 0]

    def update(frame: int):
        line_index = 0
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            points = joints[frame, [parent, child]]
            lines[line_index].set_data(points[:, 0], points[:, 2])
            lines[line_index].set_3d_properties(points[:, 1])
            line_index += 1
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[2] - radius, center[2] + radius)
        axis.set_zlim(center[1] - radius, center[1] + radius)
        axis.set_title(f"TempoGraph frame {frame}")
        return lines

    animation = FuncAnimation(figure, update, frames=motion.frames, interval=1000 / (fps or motion.fps))
    writer = PillowWriter(fps=fps or motion.fps) if output.suffix.lower() == ".gif" else FFMpegWriter(fps=fps or motion.fps)
    animation.save(output, writer=writer)
    plt.close(figure)

