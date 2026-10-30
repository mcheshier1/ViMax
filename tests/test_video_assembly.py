import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from moviepy import AudioClip, ImageClip, VideoFileClip
from PIL import Image

from utils.video_assembly import assemble_video_files


class VideoAssemblyBehaviorTests(unittest.TestCase):
    def _pattern(self, width=640, height=360, seed=17):
        rng = np.random.default_rng(seed)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # Locally distinct shapes across the whole frame make false local matches visible.
        for _ in range(240):
            x, y = rng.integers(8, width - 8), rng.integers(8, height - 8)
            color = tuple(int(c) for c in rng.integers(30, 256, size=3))
            cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 8)), color, -1)
        return image

    def _video(self, path, image, *, duration=0.8, size=None, audio=False):
        frame = image if size is None else cv2.resize(image, size)
        clip = ImageClip(frame).with_duration(duration)
        if audio:
            sound = AudioClip(lambda t: np.column_stack((0.15 * np.sin(2 * np.pi * 440 * t),
                                                          0.15 * np.sin(2 * np.pi * 440 * t))), duration=duration, fps=48000)
            clip = clip.with_audio(sound)
        clip.write_videofile(str(path), fps=12, codec="libx264", audio=audio, threads=1, logger=None)
        clip.close()

    @staticmethod
    def _fingerprint(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_bounded_anchor_correction_changes_join_geometry_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "joined.mp4"
            anchor_path = root / "anchor.png"
            image = self._pattern()
            markers = [(30, 30), (610, 30), (30, 330), (610, 330)]
            for x, y in markers:
                cv2.circle(image, (x, y), 11, (255, 255, 255), -1)
            self._video(source, image, audio=True)
            source_duration = float(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(source),
            ]))
            # An ideal keyframe differs by a small, physically plausible scale and offset.
            anchor = cv2.warpAffine(image, np.float32([[1.01, 0, -6], [0, 1.01, -4]]), (640, 360))
            Image.fromarray(anchor).save(anchor_path)
            before = self._fingerprint(source)
            first_only = assemble_video_files([{"path": str(source), "first_frame": str(anchor_path)}],
                                              root / "first-only.mp4", width=640, height=360)
            self.assertEqual(first_only["clips"][0]["status"], "unchanged")
            self.assertIn("end-to-end correction is unavailable", first_only["clips"][0]["reason"])


            report = assemble_video_files([{"path": str(source), "first_frame": str(anchor_path),
                                            "last_frame": str(anchor_path), "slot": "opening"}],
                                          output, width=640, height=360)

            self.assertEqual(report["clips"][0]["status"], "corrected")
            self.assertEqual(report["clips"][0]["duration_seconds"], source_duration)
            self.assertGreater(report["crop_zoom"], 1.0)
            self.assertEqual(self._fingerprint(source), before)
            with VideoFileClip(str(output)) as film:
                self.assertAlmostEqual(film.duration, source_duration, delta=1 / 30)
                self.assertIsNotNone(film.audio)
                self.assertEqual(tuple(film.size), (640, 360))
                frame = film.get_frame(0.2)
                zoom = report["crop_zoom"]
                for x, y in markers:
                    target_x = zoom * (1.01 * x - 6 - 320) + 320
                    target_y = zoom * (1.01 * y - 4 - 180) + 180
                    patch = frame[int(target_y)-8:int(target_y)+9, int(target_x)-8:int(target_x)+9]
                    self.assertTrue(np.any(np.all(patch > 190, axis=2)), "Each corner landmark must retain its correct orientation")

    def test_no_anchor_preserves_aspect_with_padding_and_audio_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "portraitish.mp4", root / "joined.mp4"
            self._video(source, self._pattern(480, 360), duration=0.75, audio=True)

            report = assemble_video_files([{"path": str(source)}], output)

            self.assertEqual(report["clips"][0]["status"], "unchanged")
            self.assertIn("aspect", report["clips"][0]["reason"])
            with VideoFileClip(str(output)) as film:
                self.assertAlmostEqual(film.duration, 0.75, delta=0.12)
                self.assertTrue(film.audio)
                # 4:3 source is contained, never stretched to the 16:9 output.
                frame = film.get_frame(0.2)
                self.assertTrue(np.all(frame[:, 0] < 20))
                self.assertTrue(np.all(frame[:, 1279] < 20))

    def test_identity_clip_shares_corrected_film_crop_without_being_marked_corrected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            image = self._pattern()
            self._video(source, image)
            identity = root / "identity.png"
            corrected = root / "corrected.png"
            Image.fromarray(image).save(identity)
            warped = cv2.warpAffine(image, np.float32([[1.01, 0, -6], [0, 1.01, -4]]), (640, 360))
            Image.fromarray(warped).save(corrected)

            report = assemble_video_files([
                {"path": str(source), "first_frame": str(identity), "last_frame": str(identity), "slot": "identity"},
                {"path": str(source), "first_frame": str(corrected), "last_frame": str(corrected), "slot": "corrected"},
            ], root / "joined.mp4")

            self.assertEqual([clip["status"] for clip in report["clips"]], ["unchanged", "corrected"])
            self.assertTrue(all(clip["duration_seconds"] > 0 for clip in report["clips"]))
            with VideoFileClip(str(root / "joined.mp4")) as film:
                self.assertAlmostEqual(film.duration, 1.6, delta=0.15)

    def test_portrait_anchor_baseline_is_padded_not_stretched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, anchor, output = root / "portrait.mp4", root / "portrait.png", root / "joined.mp4"
            image = self._pattern(360, 640)
            self._video(source, image)
            Image.fromarray(image).save(anchor)

            report = assemble_video_files([{"path": str(source), "first_frame": str(anchor)}], output)

            self.assertEqual(report["clips"][0]["status"], "unchanged")
            self.assertIn("end-to-end correction is unavailable", report["clips"][0]["reason"])
            with VideoFileClip(str(output)) as film:
                frame = film.get_frame(0.2)
                self.assertTrue(np.all(frame[:, :20] < 20))
                self.assertTrue(np.all(frame[:, -20:] < 20))
                self.assertGreater(int(frame[:, 400:880].max()), 200)

    def test_unrelated_or_excessive_anchor_is_reported_not_warped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._video(source, self._pattern(seed=2))
            unrelated = root / "unrelated.png"
            Image.fromarray(self._pattern(seed=99)).save(unrelated)

            result = assemble_video_files([{"path": str(source), "first_frame": str(unrelated)}], root / "unrelated-out.mp4")
            self.assertEqual(result["clips"][0]["status"], "needs_review")
            self.assertIn("reliable", result["clips"][0]["reason"])
            portrait_reference = root / "portrait-reference.png"
            Image.fromarray(self._pattern(360, 640)).save(portrait_reference)
            result = assemble_video_files([{
                "path": str(source), "first_frame": str(unrelated), "last_frame": str(portrait_reference),
            }], root / "aspect-mismatch.mp4")
            self.assertEqual(result["clips"][0]["status"], "needs_review")
            self.assertIn("different aspect", result["clips"][0]["reason"])

            image = self._pattern(seed=2)
            large = cv2.warpAffine(image, np.float32([[1.09, 0, -28], [0, 1.09, -16]]), (640, 360))
            large_path = root / "large.png"
            Image.fromarray(large).save(large_path)
            result = assemble_video_files([{"path": str(source), "first_frame": str(large_path)}], root / "large-out.mp4")
            self.assertEqual(result["clips"][0]["status"], "needs_review")
            self.assertTrue(any(word in result["clips"][0]["reason"] for word in ("bounds", "reliable")))

    def test_lighting_match_applies_constant_gain_without_mutating_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            y, x = np.mgrid[:360, :640]
            gray = (65 + ((x // 16 + y // 16) % 4) * 28).astype(np.uint8)
            reference = np.repeat(gray[:, :, None], 3, axis=2)
            darker = np.rint(reference.astype(np.float32) * 0.95).astype(np.uint8)
            first, second, output = root / "reference.mp4", root / "darker.mp4", root / "joined.mp4"
            self._video(first, reference)
            self._video(second, darker)
            hashes = [self._fingerprint(first), self._fingerprint(second)]

            report = assemble_video_files([{"path": str(first), "slot": "reference"},
                                           {"path": str(second), "slot": "darker"}],
                                          output, width=640, height=360)

            lighting = report["clips"][1]["lighting"]
            self.assertEqual(report["clips"][0]["lighting"]["status"], "unchanged")
            self.assertEqual(report["clips"][0]["lighting"]["gains"], [1.0, 1.0, 1.0])
            self.assertEqual(lighting["status"], "corrected")
            self.assertEqual(lighting["previous_slot"], "reference")
            self.assertTrue(all(1.03 < gain < 1.08 for gain in lighting["gains"]))
            with VideoFileClip(str(output)) as film:
                before = film.get_frame(0.4).mean()
                after = film.get_frame(1.2).mean()
            self.assertLess(abs(before - after), 2.0)
            self.assertEqual([self._fingerprint(first), self._fingerprint(second)], hashes)

    def test_failed_render_does_not_publish_partial_output_or_leak_staging_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.mp4", root / "out.mp4"
            self._video(source, self._pattern())
            output.write_bytes(b"previous complete output")

            with self.assertRaises(subprocess.CalledProcessError):
                assemble_video_files([{"path": str(source)}], output, codec="codec-that-does-not-exist")

            self.assertEqual(output.read_bytes(), b"previous complete output")
            self.assertEqual(list(root.glob(".out.mp4.*.tmp.mp4")), [])


if __name__ == "__main__":
    unittest.main()
