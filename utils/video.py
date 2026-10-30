import hashlib
import json
import logging
import os
import requests
import subprocess
from fractions import Fraction
from utils.retry import download_retry
from utils.video_assembly import assemble_video_files


@download_retry
def download_video(url, save_path):
    try:
        logging.info(f"Downloading video from {url} to {save_path}")

        response = requests.get(url, stream=True, timeout=(10, 300))
        response.raise_for_status()  # 检查请求是否成功
    
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Video downloaded successfully to {save_path}")
    
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        raise e



def _source_canvas(video_paths):
    width = height = 0
    fps = None
    for path in video_paths:
        result = subprocess.run(
            [
                os.environ.get("VIMAX_FFPROBE_CMD", "ffprobe"),
                "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate",
                "-of", "json", path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            raise ValueError(f"No video stream in {path}")
        stream = streams[0]
        width = max(width, int(stream["width"]))
        height = max(height, int(stream["height"]))
        if fps is None:
            try:
                fps = float(Fraction(stream.get("avg_frame_rate", "30/1")))
            except (ValueError, ZeroDivisionError):
                fps = 30.0
    width, height = (width or 1280), (height or 720)
    return width + (width % 2), height + (height % 2), (fps or 30.0)


def concatenate_video_files(video_paths, output_path, codec="libx264", preset="medium"):

    """Assemble clips without keyframe correction on the prior max-input canvas."""
    output_path = os.fspath(output_path)
    clips = [{"path": os.path.abspath(path), "slot": str(index)} for index, path in enumerate(video_paths)]
    width, height, fps = _source_canvas(video_paths)
    correction = assemble_video_files(clips, output_path, width=width, height=height, fps=fps, codec=codec, preset=preset)
    digest = hashlib.sha256()
    with open(output_path, "rb") as output:
        for chunk in iter(lambda: output.read(1024 * 1024), b""):
            digest.update(chunk)
    report_path = output_path + ".correction.json"
    temporary = report_path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as report:
            json.dump({"outputHash": digest.hexdigest(), "correction": correction}, report, ensure_ascii=False, indent=2)
        os.replace(temporary, report_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return output_path
