import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

import utils.video_assembly as video_assembly
from utils.video_assembly import prepare_rendered_ending


class RenderedEndingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def _frames(self, folder, colors, *, width=128, height=96):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, color in enumerate(colors):
            path = folder / f"frame-{index:03d}.png"
            Image.new("RGB", (width, height), color).save(path)
            paths.append(path)
        return paths

    def _video(self, path, colors, *, b_frames=True, width=128, height=96):
        frames = self._frames(self.root / (Path(path).stem + "-frames"), colors,
                              width=width, height=height)
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "6", "-i", str(frames[0].parent / "frame-%03d.png"),
            "-frames:v", str(len(frames)), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-bf", "2" if b_frames else "0", str(path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return Path(path)

    def _video_images(self, path, images, *, b_frames=False):
        folder = self.root / (Path(path).stem + "-images")
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, image in enumerate(images):
            frame_path = folder / f"frame-{index:03d}.png"
            Image.fromarray(image).save(frame_path)
            paths.append(frame_path)
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "6", "-i", str(folder / "frame-%03d.png"),
            "-frames:v", str(len(paths)), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-bf", "2" if b_frames else "0", str(path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return Path(path)

    @staticmethod
    def _pattern(seed):
        rng = np.random.default_rng(seed)
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(240):
            x, y = rng.integers(8, 632), rng.integers(8, 352)
            color = tuple(int(channel) for channel in rng.integers(30, 256, size=3))
            cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 8)), color, -1)
        return image

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _center_rgb(path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            return np.asarray(image)[image.height // 2, image.width // 2].astype(int)

    def test_extracts_actual_last_displayed_frame_not_planned_still_and_preserves_source(self):
        source = self._video(self.root / "clip.mp4", [(220, 20, 20), (20, 210, 20), (15, 25, 230)])
        planned = self.root / "planned.png"
        Image.new("RGB", (128, 96), (230, 220, 15)).save(planned)
        source_hash = self._sha256(source)

        result = prepare_rendered_ending(source, last_frame=planned)

        self.assertEqual(result["path"], str(source) + ".rendered-ending.png")
        self.assertEqual(result["clip_sha256"], source_hash)
        self.assertEqual(result["normalization"], "rendered-end-v1")
        self.assertEqual((result["width"], result["height"]), (1280, 720))
        center = self._center_rgb(result["path"])
        self.assertGreater(center[2] - center[0], 150)
        self.assertNotEqual(self._center_rgb(result["path"]).tolist(), [230, 220, 15])
        self.assertEqual(self._sha256(source), source_hash)

    def test_unreliable_first_or_single_anchor_preserves_assembly_fallback_geometry(self):
        first_image = self._pattern(3)
        last_image = self._pattern(17)
        source = self._video_images(self.root / "anchored.mp4",
                                    [first_image, self._pattern(7), last_image])
        last_anchor = self.root / "last-anchor.png"
        first_anchor = self.root / "unrelated-first.png"
        Image.fromarray(last_image).save(last_anchor)
        Image.fromarray(self._pattern(99)).save(first_anchor)

        plain = prepare_rendered_ending(source)
        inconsistent = prepare_rendered_ending(source, first_frame=first_anchor, last_frame=last_anchor)
        last_only = prepare_rendered_ending(source, last_frame=last_anchor)

        with Image.open(plain["path"]) as image:
            expected = np.asarray(image)
        for result in (inconsistent, last_only):
            with Image.open(result["path"]) as image:
                np.testing.assert_array_equal(np.asarray(image), expected)

    def test_vfr_b_frames_and_single_frame_decode_the_last_presentation_frame(self):
        frames = self._frames(self.root / "vfr-frames", [(225, 15, 15), (15, 220, 15), (15, 15, 225)])
        concat = self.root / "vfr.txt"
        concat.write_text(
            "\n".join([
                f"file '{frames[0]}'", "duration 0.04",
                f"file '{frames[1]}'", "duration 0.65",
                f"file '{frames[2]}'", "duration 0.08",
                f"file '{frames[2]}'", "",
            ]),
            encoding="utf-8",
        )
        vfr = self.root / "vfr.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat), "-fps_mode", "vfr",
            "-c:v", "libx264", "-bf", "2", "-pix_fmt", "yuv420p", str(vfr),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        result = prepare_rendered_ending(vfr)
        pixel = self._center_rgb(result["path"])
        self.assertGreater(pixel[2], pixel[0] + 120)
        self.assertGreater(pixel[2], pixel[1] + 120)

        one_frame = self._video(self.root / "one-frame.mp4", [(20, 30, 225)], b_frames=False)
        one_result = prepare_rendered_ending(one_frame)
        self.assertGreater(self._center_rgb(one_result["path"])[2], 180)

    def test_sar_and_rotation_are_applied_before_canonical_padding(self):
        frames = self._frames(self.root / "rotated-frames", [(30, 40, 220)], width=128, height=96)
        source = self.root / "rotated.mp4"
        encoded = self.root / "unrotated.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "1", "-i", str(frames[0]), "-vf", "setsar=2/1",
            "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(encoded),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-display_rotation:v:0", "90", "-i", str(encoded), "-c", "copy", str(source),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        result = prepare_rendered_ending(source)
        with Image.open(result["path"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        content = np.any(pixels > 40, axis=2)
        ys, xs = np.where(content)
        self.assertGreater(ys.max() - ys.min(), 600)
        self.assertLess(xs.max() - xs.min(), 500)

    def test_cache_reuse_and_source_anchor_normalization_and_png_invalidation(self):
        source = self._video(self.root / "cache.mp4", [(230, 20, 20), (20, 20, 230)])
        first_anchor = self.root / "first.png"
        last_anchor = self.root / "last.png"
        Image.new("RGB", (128, 96), (10, 180, 10)).save(first_anchor)
        Image.new("RGB", (128, 96), (200, 180, 10)).save(last_anchor)
        initial = prepare_rendered_ending(source, first_frame=first_anchor, last_frame=last_anchor)

        with mock.patch.object(video_assembly, "_rendered_ending_frame", side_effect=AssertionError("cache miss")):
            self.assertEqual(prepare_rendered_ending(source, first_frame=first_anchor,
                                                     last_frame=last_anchor), initial)

        Image.new("RGB", (1280, 720), (0, 0, 0)).save(initial["path"])
        tampered_hash = self._sha256(initial["path"])
        with mock.patch.object(video_assembly, "_rendered_ending_frame",
                               wraps=video_assembly._rendered_ending_frame) as decode:
            repaired = prepare_rendered_ending(source, first_frame=first_anchor, last_frame=last_anchor)
        decode.assert_called_once()
        self.assertNotEqual(tampered_hash, repaired["sha256"])
        self.assertEqual(repaired["sha256"], self._sha256(repaired["path"]))

        old_first_hash = repaired["normalization_inputs"]["first_frame_sha256"]
        Image.new("RGB", (128, 96), (10, 180, 200)).save(first_anchor)
        with mock.patch.object(video_assembly, "_rendered_ending_frame",
                               wraps=video_assembly._rendered_ending_frame) as decode:
            anchor_changed = prepare_rendered_ending(source, first_frame=first_anchor, last_frame=last_anchor)
        decode.assert_called_once()
        self.assertNotEqual(anchor_changed["normalization_inputs"]["first_frame_sha256"], old_first_hash)

        old_normalization = anchor_changed["normalization"]
        with mock.patch.object(video_assembly, "RENDERED_ENDING_NORMALIZATION", "rendered-end-test"):
            with mock.patch.object(video_assembly, "_rendered_ending_frame",
                                   wraps=video_assembly._rendered_ending_frame) as decode:
                normalized_again = prepare_rendered_ending(source, first_frame=first_anchor, last_frame=last_anchor)
            decode.assert_called_once()
        self.assertNotEqual(normalized_again["normalization"], old_normalization)

        replacement = self._video(self.root / "replacement.mp4", [(10, 220, 20), (210, 20, 210)])
        replacement_hash = self._sha256(replacement)
        shutil.copyfile(replacement, source)
        with mock.patch.object(video_assembly, "_rendered_ending_frame",
                               wraps=video_assembly._rendered_ending_frame) as decode:
            replaced = prepare_rendered_ending(source, first_frame=first_anchor, last_frame=last_anchor)
        decode.assert_called_once()
        self.assertEqual(replaced["clip_sha256"], replacement_hash)
        self.assertGreater(self._center_rgb(replaced["path"])[0], 130)
        self.assertGreater(self._center_rgb(replaced["path"])[2], 130)

    def test_failed_decode_does_not_fall_back_to_anchor_or_publish_cache(self):
        source = self.root / "broken.mp4"
        source.write_bytes(b"not a video")
        planned = self.root / "planned.png"
        Image.new("RGB", (128, 96), (220, 20, 20)).save(planned)

        with self.assertRaises(ValueError):
            prepare_rendered_ending(source, last_frame=planned)
        self.assertFalse(Path(str(source) + ".rendered-ending.png").exists())
        self.assertFalse(Path(str(source) + ".rendered-ending.json").exists())

    def test_rendered_ending_json_stdin_cli_returns_metadata(self):
        source = self._video(self.root / "cli.mp4", [(20, 20, 200)])
        script = Path(video_assembly.__file__).resolve()
        completed = subprocess.run(
            [sys.executable, str(script), "--rendered-ending", str(source)],
            input=json.dumps({"width": 320, "height": 180}), text=True,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        result = json.loads(completed.stdout)
        self.assertEqual((result["width"], result["height"]), (320, 180))
        self.assertTrue(Path(result["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
