import unittest

from tempograph.data.pseudo_graph import build_pseudo_choreography_graph
from tempograph.synthetic import synthetic_motion, synthetic_music


class GraphTest(unittest.TestCase):
    def test_pseudo_graph_covers_sequence(self):
        motion = synthetic_motion(frames=180)
        music = synthetic_music(frames=180)
        graph = build_pseudo_choreography_graph(
            motion,
            music,
            min_segment_frames=30,
            max_segment_frames=90,
        )
        graph.validate()
        self.assertEqual(graph.nodes[0].start_frame, 0)
        self.assertEqual(graph.nodes[-1].end_frame, 180)
        self.assertGreaterEqual(len(graph.nodes), 2)
        self.assertGreaterEqual(len(graph.edges), len(graph.nodes) - 1)


if __name__ == "__main__":
    unittest.main()

