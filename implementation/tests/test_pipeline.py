import unittest

from tempograph.evaluation import boundary_continuity_metrics
from tempograph.pipeline import TempoGraphPipeline
from tempograph.synthetic import synthetic_motion, synthetic_music


class PipelineTest(unittest.TestCase):
    def test_end_to_end_gap_fill(self):
        motion = synthetic_motion(frames=180)
        music = synthetic_music(frames=180)
        result = TempoGraphPipeline().fill_gap(motion, music, 75, 105)
        self.assertGreater(len(result.graph.nodes), 0)
        self.assertGreaterEqual(result.plan.length, 8)
        self.assertEqual(result.motion.frames, 75 + result.plan.length + (180 - 105))
        self.assertTrue(result.transition.metadata.get("contact_corrected"))
        self.assertIsNotNone(result.motion.local_joint_positions)
        metrics = boundary_continuity_metrics(
            motion.slice(60, 75), result.transition, motion.slice(105, 120)
        )
        self.assertLess(metrics["contact_jump"], 1e-6)

    def test_optional_models_fail_cleanly_without_torch(self):
        from tempograph.models import load_torch_models

        try:
            models = load_torch_models()
        except RuntimeError as error:
            self.assertIn("PyTorch", str(error))
        else:
            self.assertIn("GraphPlanner", models)


if __name__ == "__main__":
    unittest.main()
