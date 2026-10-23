import unittest

import numpy as np

from tempograph.geometry.hermite import cubic_hermite, cubic_hermite_velocity
from tempograph.geometry.rotations import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    so3_exp,
    so3_log,
)


class GeometryTest(unittest.TestCase):
    def test_rotation_6d_round_trip(self):
        vectors = np.asarray([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]], dtype=np.float32)
        matrices = so3_exp(vectors)
        rebuilt = rotation_6d_to_matrix(matrix_to_rotation_6d(matrices))
        np.testing.assert_allclose(rebuilt, matrices, atol=1e-5)

    def test_so3_log_exp_round_trip(self):
        vectors = np.asarray([[0.1, 0.2, -0.1], [0.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(so3_log(so3_exp(vectors)), vectors, atol=1e-5)

    def test_hermite_endpoints_and_velocities(self):
        start = np.asarray([0.0, 0.0, 0.0])
        end = np.asarray([2.0, 1.0, -1.0])
        start_velocity = np.asarray([0.5, 0.0, 0.0])
        end_velocity = np.asarray([0.0, 0.25, 0.0])
        phase = np.asarray([0.0, 1.0])
        curve = cubic_hermite(start, end, start_velocity, end_velocity, 2.0, phase)
        velocity = cubic_hermite_velocity(start, end, start_velocity, end_velocity, 2.0, phase)
        np.testing.assert_allclose(curve[0], start, atol=1e-6)
        np.testing.assert_allclose(curve[-1], end, atol=1e-6)
        np.testing.assert_allclose(velocity[0], start_velocity, atol=1e-6)
        np.testing.assert_allclose(velocity[-1], end_velocity, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

