import unittest

import numpy as np

from tempograph.boundary.bridge import AnalyticBoundaryBridge
from tempograph.boundary.features import boundary_conflict, gated_rigid_penalty
from tempograph.boundary.planner import AnalyticBoundaryPlanner
from tempograph.synthetic import synthetic_motion


class BoundaryTest(unittest.TestCase):
    def setUp(self):
        motion = synthetic_motion(frames=120)
        self.left = motion.slice(30, 45)
        self.right = motion.slice(75, 90)

    def test_conflict_vector(self):
        conflict = boundary_conflict(self.left, self.right)
        self.assertEqual(conflict.shape, (5,))
        self.assertTrue(np.all(conflict >= 0))

    def test_plan_and_bridge_endpoints(self):
        plan = AnalyticBoundaryPlanner().plan(self.left, self.right, 0.5)
        transition = AnalyticBoundaryBridge().generate(self.left, self.right, plan)
        self.assertEqual(transition.frames, plan.length)
        np.testing.assert_allclose(transition.root_positions[0], self.left.root_positions[-1], atol=1e-6)
        np.testing.assert_allclose(transition.root_positions[-1], self.right.root_positions[0], atol=1e-6)
        np.testing.assert_allclose(transition.contacts[0], self.left.contacts[-1], atol=1e-6)
        np.testing.assert_allclose(transition.contacts[-1], self.right.contacts[0], atol=1e-6)

    def test_rigid_gate_ignores_low_reference_energy(self):
        penalty = gated_rigid_penalty(
            np.ones(10) * 10,
            np.zeros(10),
            np.zeros(10),
            cold_threshold=0.1,
            hot_threshold=1.0,
            root_threshold=0.1,
            articulation_threshold=1.0,
        )
        self.assertEqual(penalty, 0.0)


if __name__ == "__main__":
    unittest.main()

