import math
import unittest

import numpy as np

from gui.viewers.transform_gizmo import (
    GizmoInteractionState,
    TransformGizmo,
)


def project_isometric(x, y, z):
    return 200.0 + 100.0 * x + 30.0 * y, 200.0 - 100.0 * z + 20.0 * y


def unused_ray(sx, sy):
    return np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0])


class TransformGizmoTests(unittest.TestCase):
    def test_center_sphere_picks_and_drags_freely_in_camera_plane(self):
        gizmo = TransformGizmo((0.0, 0.0, 0.5))

        def project(x, y, z):
            return 200.0 + 100.0 * x, 200.0 + 100.0 * y

        def ray(sx, sy):
            return (
                np.array([(sx - 200.0) / 100.0, (sy - 200.0) / 100.0, 5.0]),
                np.array([0.0, 0.0, -1.0]),
            )

        state, handle = gizmo.pick(200.0, 200.0, project)
        self.assertEqual(state, GizmoInteractionState.HOVER_TRANSLATE_FREE)
        self.assertEqual(handle, "free")
        original_quaternion = gizmo.quaternion.copy()
        self.assertTrue(gizmo.begin_drag(200.0, 200.0, project, ray))
        position, quaternion = gizmo.drag(220.0, 210.0, project, ray)
        np.testing.assert_allclose(position, [0.2, 0.1, 0.5], atol=1e-8)
        np.testing.assert_allclose(quaternion, original_quaternion)

    def test_x_arrow_pick_and_miss(self):
        gizmo = TransformGizmo((0.0, 0.0, 0.0))
        endpoint = project_isometric(gizmo.arrow_length, 0.0, 0.0)
        state, axis = gizmo.pick(*endpoint, project_isometric)
        self.assertEqual(state, GizmoInteractionState.HOVER_TRANSLATE_X)
        self.assertEqual(axis, "x")
        self.assertEqual(
            gizmo.pick(20.0, 20.0, project_isometric),
            (GizmoInteractionState.NONE, None),
        )

    def test_each_translation_changes_only_selected_axis(self):
        cases = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        for axis, vector in cases.items():
            with self.subTest(axis=axis):
                gizmo = TransformGizmo((0.0, 0.0, 0.5))
                endpoint = project_isometric(*(gizmo.position + gizmo.arrow_length * vector))
                self.assertTrue(gizmo.begin_drag(*endpoint, project_isometric, unused_ray))
                origin_screen = np.array(project_isometric(*gizmo.position))
                endpoint_screen = np.array(endpoint)
                drag_screen = endpoint_screen + 20.0 * (
                    endpoint_screen - origin_screen
                ) / np.linalg.norm(endpoint_screen - origin_screen)
                position, _ = gizmo.drag(
                    *drag_screen, project_isometric, unused_ray
                )
                changed = np.abs(position - np.array([0.0, 0.0, 0.5])) > 1e-8
                expected = np.array([axis == "x", axis == "y", axis == "z"])
                np.testing.assert_array_equal(changed, expected)

    def test_drag_shows_only_the_selected_handle_until_release(self):
        gizmo = TransformGizmo((0.0, 0.0, 0.0))
        self.assertEqual(
            gizmo.visible_handles(),
            (True, ("x", "y", "z"), ()),
        )

        endpoint = project_isometric(gizmo.arrow_length, 0.0, 0.0)
        self.assertTrue(
            gizmo.begin_drag(*endpoint, project_isometric, unused_ray)
        )
        self.assertEqual(gizmo.visible_handles(), (False, ("x",), ()))

        gizmo.end_drag()
        self.assertEqual(
            gizmo.visible_handles(),
            (True, ("x", "y", "z"), ()),
        )

    def test_mode_filters_visible_and_pickable_handles(self):
        gizmo = TransformGizmo((0.0, 0.0, 0.0))
        self.assertEqual(gizmo.visible_handles(), (True, ("x", "y", "z"), ()))
        diagonal = gizmo.ring_radius / math.sqrt(2.0)
        ring_point = project_isometric(0.0, -diagonal, -diagonal)
        self.assertEqual(
            gizmo.pick(*ring_point, project_isometric),
            (GizmoInteractionState.NONE, None),
        )

        gizmo.set_mode("rotate")
        self.assertEqual(gizmo.visible_handles(), (False, (), ("x", "y", "z")))
        state, axis = gizmo.pick(*ring_point, project_isometric)
        self.assertEqual(state, GizmoInteractionState.HOVER_ROTATE_X)
        self.assertEqual(axis, "x")

    def test_screen_scale_keeps_dimensions_in_practical_bounds(self):
        gizmo = TransformGizmo((0.0, 0.0, 0.0))
        gizmo.set_screen_scale(0.002)
        self.assertAlmostEqual(gizmo.arrow_length, 0.192)
        self.assertAlmostEqual(gizmo.ring_radius, 0.148)
        self.assertEqual(gizmo.pick_tolerance_pixels, 11.0)

        gizmo.set_screen_scale(0.5)
        self.assertEqual(gizmo.arrow_length, 0.55)
        self.assertEqual(gizmo.sphere_radius, 0.08)

    def test_translation_snap_and_fine_modifiers(self):
        gizmo = TransformGizmo((0.0, 0.0, 0.5))
        endpoint = project_isometric(gizmo.arrow_length, 0.0, 0.5)
        self.assertTrue(gizmo.begin_drag(*endpoint, project_isometric, unused_ray))

        origin_screen = np.array(project_isometric(*gizmo.position))
        endpoint_screen = np.array(endpoint)
        screen_axis = endpoint_screen - origin_screen
        unit_screen_axis = screen_axis / np.linalg.norm(screen_axis)
        position, _ = gizmo.drag(
            *(endpoint_screen + 2.6 * unit_screen_axis),
            project_isometric,
            unused_ray,
            snap=True,
        )
        np.testing.assert_allclose(position, [0.03, 0.0, 0.5], atol=1e-8)
        self.assertEqual(gizmo.drag_status(), "X +0.030 m")

        gizmo.end_drag()
        gizmo = TransformGizmo((0.0, 0.0, 0.5))
        endpoint = project_isometric(gizmo.arrow_length, 0.0, 0.5)
        origin_screen = np.array(project_isometric(*gizmo.position))
        endpoint_screen = np.array(endpoint)
        screen_axis = endpoint_screen - origin_screen
        unit_screen_axis = screen_axis / np.linalg.norm(screen_axis)
        self.assertTrue(gizmo.begin_drag(*endpoint, project_isometric, unused_ray))
        position, _ = gizmo.drag(
            *(endpoint_screen + 20.0 * unit_screen_axis),
            project_isometric,
            unused_ray,
            fine=True,
        )
        np.testing.assert_allclose(position, [0.05, 0.0, 0.5], atol=1e-8)

    def test_rotation_ring_constrains_quaternion_axis(self):
        camera_setups = {
            "x": (
                lambda x, y, z: (200 + 100 * y, 200 - 100 * z),
                lambda sx, sy: (
                    np.array([1.0, (sx - 200) / 100, -(sy - 200) / 100]),
                    np.array([-1.0, 0.0, 0.0]),
                ),
                (214.142, 185.858), (185.858, 185.858), 1,
            ),
            "y": (
                lambda x, y, z: (200 + 100 * x, 200 - 100 * z),
                lambda sx, sy: (
                    np.array([(sx - 200) / 100, 1.0, -(sy - 200) / 100]),
                    np.array([0.0, -1.0, 0.0]),
                ),
                (214.142, 185.858), (185.858, 185.858), 2,
            ),
            "z": (
                lambda x, y, z: (200 + 100 * x, 200 + 100 * y),
                lambda sx, sy: (
                    np.array([(sx - 200) / 100, (sy - 200) / 100, 1.0]),
                    np.array([0.0, 0.0, -1.0]),
                ),
                (214.142, 214.142), (185.858, 214.142), 3,
            ),
        }
        for axis, (project, ray, start, end, quaternion_index) in camera_setups.items():
            with self.subTest(axis=axis):
                gizmo = TransformGizmo((0.0, 0.0, 0.0))
                gizmo.set_mode("rotate")
                self.assertTrue(gizmo.begin_drag(*start, project, ray))
                self.assertIn("ROTATE", gizmo.state.name)
                self.assertEqual(
                    gizmo.visible_handles(), (False, (), (axis,))
                )
                _, quaternion = gizmo.drag(*end, project, ray)
                other_vector_parts = np.delete(quaternion[1:], quaternion_index - 1)
                np.testing.assert_allclose(other_vector_parts, 0.0, atol=1e-7)
                self.assertAlmostEqual(abs(quaternion[quaternion_index]), math.sqrt(0.5), places=5)
                gizmo.end_drag()
                self.assertEqual(
                    gizmo.visible_handles(),
                    (False, (), ("x", "y", "z")),
                )


if __name__ == "__main__":
    unittest.main()
