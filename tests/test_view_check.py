"""Combine must refuse two videos that show the actor from the same or from opposite sides."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import combine  # noqa: E402
import synthetic_combine  # noqa: E402


def run(angle, **kwargs):
    take = synthetic_combine.make_take(seed=0, side_angle=angle, swap_bursts=0)
    return combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                    take["visF"], take["visS"], **kwargs)[1]


class ViewCheck(unittest.TestCase):
    def test_side_views_are_combined(self):
        for angle in (90.0, 70.0, 120.0):
            report = run(angle)
            self.assertTrue(report["combined"], angle)
            self.assertLess(abs(report["view_slope"]), combine.SAME_PLANE_SLOPE)

    def test_opposite_and_same_side_are_refused(self):
        for angle, word in ((170.0, "opposite sides"), (10.0, "the same side")):
            report = run(angle)
            self.assertFalse(report["combined"], angle)
            self.assertIn(word, report["reason"])

    def test_given_angle_is_not_second_guessed(self):
        self.assertTrue(run(170.0, side_angle=140.0)["combined"])


if __name__ == "__main__":
    unittest.main()
