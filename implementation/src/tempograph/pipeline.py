"""End-to-end orchestration shared by CLI, tests, and future training code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import AnalyticBoundaryBridge, AnalyticBoundaryPlanner, contact_aware_root_correction
from .boundary.bridge import concatenate_with_transition
from .data.music import beat_interval_near
from .data.pseudo_graph import build_pseudo_choreography_graph
from .types import BoundaryPlan, ChoreographyGraph, MotionSequence, MusicFeatures


@dataclass
class PipelineResult:
    graph: ChoreographyGraph
    plan: BoundaryPlan
    transition: MotionSequence
    motion: MotionSequence
    gap_start: int
    gap_end: int


class TempoGraphPipeline:
    def __init__(
        self,
        context_frames: int = 15,
        min_transition_frames: int = 8,
        max_transition_frames: int = 60,
    ) -> None:
        self.context_frames = context_frames
        self.planner = AnalyticBoundaryPlanner(min_transition_frames, max_transition_frames)
        self.bridge = AnalyticBoundaryBridge()

    def analyze(self, motion: MotionSequence, music: MusicFeatures) -> ChoreographyGraph:
        return build_pseudo_choreography_graph(
            motion,
            music,
            min_segment_frames=max(10, int(round(motion.fps))),
            max_segment_frames=max(30, int(round(motion.fps * 8))),
        )

    def fill_gap(
        self,
        motion: MotionSequence,
        music: MusicFeatures,
        gap_start: int,
        gap_end: int,
        edge_type: str = "transition",
    ) -> PipelineResult:
        if gap_start < self.context_frames or gap_end + self.context_frames > motion.frames:
            raise ValueError("gap does not leave enough boundary context")
        if gap_end <= gap_start:
            raise ValueError("gap_end must be after gap_start")
        graph = self.analyze(motion, music)
        left_context = motion.slice(gap_start - self.context_frames, gap_start)
        right_context = motion.slice(gap_end, gap_end + self.context_frames)
        seam_time = gap_start / motion.fps
        plan = self.planner.plan(
            left_context,
            right_context,
            beat_interval_near(seam_time, music.beats),
            edge_type,
        )
        transition = contact_aware_root_correction(
            self.bridge.generate(left_context, right_context, plan),
            iterations=8,
        )
        completed = concatenate_with_transition(
            motion.slice(0, gap_start),
            transition,
            motion.slice(gap_end, motion.frames),
        )
        return PipelineResult(graph, plan, transition, completed, gap_start, gap_end)


def graph_to_dict(graph: ChoreographyGraph) -> dict[str, object]:
    return {
        "total_frames": graph.total_frames,
        "nodes": [
            {
                "index": node.index,
                "start_frame": node.start_frame,
                "end_frame": node.end_frame,
                "role": node.role,
                "semantic": node.semantic,
                "keyframes": node.keyframes,
                "confidence": node.confidence,
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type,
                "confidence": edge.confidence,
            }
            for edge in graph.edges
        ],
    }


def plan_to_dict(plan: BoundaryPlan) -> dict[str, object]:
    return {
        "length": plan.length,
        "conflict": plan.conflict.tolist(),
        "energy_min": float(np.min(plan.energy)),
        "energy_max": float(np.max(plan.energy)),
        "contact_switches": int(np.sum(np.diff(np.argmax(plan.contact_probabilities[:, :2], axis=-1)) != 0)),
    }
