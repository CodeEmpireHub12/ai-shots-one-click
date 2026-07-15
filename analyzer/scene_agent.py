"""Scene Agent for Project Raven.

Uses PySceneDetect to find scene boundaries and OpenCV to compute rough
motion and brightness features for each detected scene.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "scene_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "scene_detection_threshold": 27.0,
    "scene_sample_seconds": 0.5,
}


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _get_video_metadata(video_path: Path) -> Tuple[float, float, int]:
    """Return fps, duration, and frame count from OpenCV."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = float(frame_count / fps) if fps > 0 else 0.0
        return fps, duration, frame_count
    finally:
        capture.release()


def _detect_scenes(video_path: Path, threshold: float, duration: float) -> List[Dict[str, float]]:
    """Detect scene boundaries with PySceneDetect ContentDetector."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=float(threshold)))
        scene_manager.detect_scenes(video=video)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for start_time, end_time in scene_list:
            start = float(start_time.get_seconds())
            end = float(end_time.get_seconds())
            if end > start:
                scenes.append({"start": start, "end": end})

        if scenes:
            return scenes

        logger.info("PySceneDetect found no cuts. Falling back to one full-length scene.")
        return [{"start": 0.0, "end": float(duration)}] if duration > 0 else []

    except Exception as exc:
        logger.exception("PySceneDetect failed; falling back to one full-length scene.")
        return [{"start": 0.0, "end": float(duration)}] if duration > 0 else []


def _compute_scene_features(
    video_path: Path,
    scene: Dict[str, float],
    fps: float,
    sample_seconds: float,
) -> Dict[str, float]:
    """Compute rough motion score and average brightness for one scene."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        start = float(scene["start"])
        end = float(scene["end"])
        safe_fps = fps if fps > 0 else 30.0
        sample_step = max(1, int(round(float(sample_seconds) * safe_fps)))
        start_frame = max(0, int(round(start * safe_fps)))
        end_frame = max(start_frame + 1, int(round(end * safe_fps)))

        brightness_values = []
        motion_values = []
        previous_gray = None

        for frame_index in range(start_frame, end_frame, sample_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(float(np.mean(gray) / 255.0))

            if previous_gray is not None:
                difference = cv2.absdiff(gray, previous_gray)
                motion_values.append(float(np.mean(difference) / 255.0))

            previous_gray = gray

        brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
        motion_score = float(np.mean(motion_values)) if motion_values else 0.0

        return {
            "start": start,
            "end": end,
            "motion_score": motion_score,
            "brightness": brightness,
        }
    finally:
        capture.release()


def analyze_scenes(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze scenes in a video.

    Expected input:
    {"video_path": "/path/to/source.mp4"}
    """
    try:
        video_path_value = input_data.get("video_path")
        if not video_path_value:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="video_path is required.",
            )

        video_path = Path(str(video_path_value)).expanduser().resolve()
        if not video_path.exists() or not video_path.is_file():
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error=f"Video file does not exist: {video_path}",
            )

        config = load_config()
        threshold = float(_config_value(config, "scene_detection_threshold"))
        sample_seconds = float(_config_value(config, "scene_sample_seconds"))

        fps, duration, frame_count = _get_video_metadata(video_path)
        if duration <= 0.0 or frame_count <= 0:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="Could not read valid video duration/frame count.",
            )

        raw_scenes = _detect_scenes(video_path, threshold, duration)
        scenes = [
            _compute_scene_features(video_path, scene, fps, sample_seconds)
            for scene in raw_scenes
        ]

        data = {"scenes": scenes}
        logger.info("Scene analysis completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Scene Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m raven.analyzer.scene_agent <video_path>")
        raise SystemExit(1)

    print(json.dumps(analyze_scenes({"video_path": sys.argv[1]}), indent=2))
