import unittest

import numpy as np

from utils.video_lighting import estimate_gains


class VideoLightingEstimatorTests(unittest.TestCase):
    @staticmethod
    def _samples(image):
        return [image.copy() for _ in range(3)]

    def _scene(self):
        y, x = np.mgrid[:180, :320]
        gray = (65 + ((x // 16 + y // 16) % 4) * 28).astype(np.uint8)
        # Neutral stable texture provides broad, unsaturated candidates.
        return np.repeat(gray[:, :, None], 3, axis=2)

    def test_recovers_constant_gain_and_improves_stable_background_match(self):
        reference = self._scene()
        darker = np.rint(reference.astype(np.float32) * 0.95).astype(np.uint8)
        result = estimate_gains(self._samples(reference), self._samples(darker))
        self.assertEqual(result["status"], "corrected")
        self.assertTrue(all(1.03 < gain < 1.08 for gain in result["gains"]))
        before = np.mean(np.abs(reference.astype(float) - darker.astype(float)))
        corrected = np.clip(darker.astype(float) * np.asarray(result["gains"]), 0, 255)
        after = np.mean(np.abs(reference.astype(float) - corrected))
        self.assertLess(after, before * 0.25)
        self.assertGreater(result["stable_fraction"], 0.5)

    def test_camera_cut_and_unstable_scene_are_skipped(self):
        reference = self._scene()
        cut = np.random.default_rng(21).integers(35, 220, size=reference.shape, dtype=np.uint8)
        result = estimate_gains(self._samples(reference), self._samples(cut))
        self.assertEqual(result["status"], "needs_review")
        self.assertRegex(result["reason"], "Structural mismatch|stable|coverage")

        moving = [np.roll(reference, shift, axis=1) for shift in (0, 30, 60)]
        result = estimate_gains(self._samples(reference), moving)
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(result["reason"])

    def test_excessive_difference_is_not_clamped_into_a_correction(self):
        reference = self._scene()
        much_darker = np.rint(reference.astype(np.float32) * 0.75).astype(np.uint8)
        result = estimate_gains(self._samples(reference), self._samples(much_darker))
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("Brightness difference", result["reason"])
        self.assertEqual(result["gains"], [1.0, 1.0, 1.0])


    def test_excluded_highlights_cannot_be_clipped_by_the_gain(self):
        reference = self._scene()
        darker = np.rint(reference.astype(np.float32) * 0.95).astype(np.uint8)
        reference[20:40, 20:40] = 246
        darker[20:40, 20:40] = 246
        result = estimate_gains(self._samples(reference), self._samples(darker))
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("clip more than 0.1%", result["reason"])

    def test_large_changed_region_cannot_justify_harming_an_unchanged_background(self):
        reference = self._scene()
        for height in (25, 60):
            with self.subTest(background_height=height):
                darker = np.rint(reference.astype(np.float32) * 0.95).astype(np.uint8)
                # Even a narrow background band must not be hidden by a majority
                # vote from the darker foreground, or by coarse validation cells.
                darker[:height] = reference[:height]
                result = estimate_gains(self._samples(reference), self._samples(darker))
                self.assertEqual(result["status"], "needs_review")
                self.assertEqual(result["gains"], [1.0, 1.0, 1.0])

if __name__ == "__main__":
    unittest.main()
