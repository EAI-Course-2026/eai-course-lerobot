#!/usr/bin/env python

import unittest

import cv2
import numpy as np

from hand_eye_calibration import (
    HandEyeSample,
    invert_transform,
    rotation_angle_degrees,
    solve_hand_eye,
    transform_from_rt,
)


def transform_from_rotation_vector(
    rotation_vector: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    return transform_from_rt(rotation, np.asarray(translation, dtype=np.float64))


class HandEyeCalibrationTest(unittest.TestCase):
    def test_inverse_transform(self) -> None:
        transform = transform_from_rotation_vector((0.2, -0.1, 0.3), (0.1, -0.2, 0.4))
        np.testing.assert_allclose(transform @ invert_transform(transform), np.eye(4), atol=1e-12)

    def test_recovers_known_eye_in_hand_transform(self) -> None:
        expected_gripper_T_camera = transform_from_rotation_vector(
            (0.18, -0.12, 0.08), (0.035, -0.012, 0.052)
        )
        fixed_base_T_target = transform_from_rotation_vector(
            (-0.08, 0.14, -0.20), (0.31, 0.04, 0.13)
        )
        robot_poses = [
            ((0.00, 0.00, 0.00), (0.18, -0.05, 0.22)),
            ((0.25, 0.02, -0.04), (0.20, -0.08, 0.24)),
            ((-0.18, 0.20, 0.03), (0.22, -0.03, 0.20)),
            ((0.08, -0.24, 0.16), (0.19, 0.01, 0.23)),
            ((-0.22, -0.12, 0.18), (0.24, -0.06, 0.19)),
            ((0.15, 0.19, -0.20), (0.17, -0.02, 0.26)),
            ((-0.12, 0.28, 0.12), (0.23, 0.02, 0.21)),
            ((0.30, -0.10, -0.14), (0.21, -0.09, 0.18)),
            ((-0.28, 0.05, -0.16), (0.16, -0.04, 0.25)),
            ((0.10, 0.30, 0.20), (0.25, -0.01, 0.22)),
            ((-0.16, -0.26, -0.08), (0.18, 0.03, 0.20)),
            ((0.22, -0.20, 0.24), (0.22, -0.05, 0.27)),
            ((-0.24, 0.16, 0.25), (0.20, 0.00, 0.17)),
            ((0.27, 0.21, 0.05), (0.26, -0.07, 0.23)),
            ((-0.20, -0.18, -0.22), (0.17, -0.08, 0.19)),
        ]
        samples = []
        for index, (rotation_vector, translation) in enumerate(robot_poses):
            base_T_gripper = transform_from_rotation_vector(rotation_vector, translation)
            camera_T_target = (
                invert_transform(expected_gripper_T_camera)
                @ invert_transform(base_T_gripper)
                @ fixed_base_T_target
            )
            samples.append(
                HandEyeSample(
                    name=f"pose_{index:02d}",
                    image="",
                    joint_degrees=(0.0, 0.0, 0.0, 0.0, 0.0),
                    base_T_gripper=base_T_gripper,
                    camera_T_target=camera_T_target,
                    pnp_reprojection_error_px=0.0,
                )
            )

        result = solve_hand_eye(samples, method="park")

        translation_error = np.linalg.norm(
            result.gripper_T_camera[:3, 3] - expected_gripper_T_camera[:3, 3]
        )
        rotation_error = rotation_angle_degrees(
            result.gripper_T_camera[:3, :3].T @ expected_gripper_T_camera[:3, :3]
        )
        self.assertLess(translation_error, 1e-8)
        self.assertLess(rotation_error, 1e-5)
        np.testing.assert_allclose(result.base_T_target, fixed_base_T_target, atol=1e-8)
        np.testing.assert_allclose(result.board_origin_rmse_mm, 0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
