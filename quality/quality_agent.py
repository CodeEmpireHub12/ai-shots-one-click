"""Quality Agent for Project Raven.

Validates a rendered short with deterministic checks:
- black frame sampling
- duration sanity check
- audio presence / non-silence check
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import librosa
import numpy as np

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "quality_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "ffprobe_path": "ffprobe",
    "quality_black_frame_threshold": 0.08,
    "quality_black_frame_fail_ratio": 0.6,
    "quality_frame_sample_count": 8,
    "quality_audio_energy_threshold": 0.005,
}


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _resolve_tool(config_value: str, fallback_name: str) -> Optional[str]:
    configured = Path(str(config_value)).expanduser()
    if configured.exists():
        return str(configured)
    from_config_name = shutil.which(str(config_value))
    if from_config_name:
        return from_config_name
    return shutil.which(fallback_name)


def _probe_duration(video_path: Path, ffprobe_path: str) -> float:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe duration check failed.")
    data = json.loads(completed.stdout)
    return float((data.get("format") or {}).get("duration") or 0.0)


def _black_frame_check(video_path: Path, sample_count: int, black_threshold: float, fail_ratio: float) -> Dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open rendered clip: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return {"status": "fail", "sampled_frames": 0, "black_ratio": 1.0}

        sample_count = max(1, min(int(sample_count), frame_count))
        indices = np.linspace(0, frame_count - 1, sample_count).astype(int)
        black_frames = 0
        sampled_frames = 0

        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray) / 255.0)
            sampled_frames += 1
            if brightness <= black_threshold:
                black_frames += 1

        if sampled_frames == 0:
            return {"status": "fail", "sampled_frames": 0, "black_ratio": 1.0}

        black_ratio = float(black_frames / sampled_frames)
        return {
            "status": "fail" if black_ratio >= fail_ratio else "pass",
            "sampled_frames": int(sampled_frames),
            "black_ratio": black_ratio,
        }
    finally:
        capture.release()


def _audio_check(video_path: Path, energy_threshold: float) -> Dict[str, Any]:
    try:
        audio, _sample_rate = librosa.load(str(video_path), sr=16000, mono=True)
        if audio.size == 0:
            return {"status": "fail", "energy": 0.0}
        energy = float(np.sqrt(np.mean(np.square(audio))))
        return {"status": "pass" if energy > energy_threshold else "fail", "energy": energy}
    except Exception as exc:
        logger.warning("Audio check failed while reading rendered clip: %s", exc)
        return {"status": "fail", "energy": 0.0, "error": str(exc)}


def check_quality(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate rendered clip quality.

    Expected input:
    {
      "rendered_clip_path": "/path/to/final.mp4",
      "expected_duration": 20.0
    }
    """
    try:
        rendered_clip_path_value = input_data.get("rendered_clip_path") or input_data.get("output_path")
        if not rendered_clip_path_value:
            return build_agent_response(AGENT_NAME, "failed", {}, 0.0, "rendered_clip_path is required.")

        rendered_clip_path = Path(str(rendered_clip_path_value)).expanduser().resolve()
        if not rendered_clip_path.exists() or not rendered_clip_path.is_file():
            return build_agent_response(
                AGENT_NAME,
                "failed",
                {},
                0.0,
                f"Rendered clip does not exist: {rendered_clip_path}",
            )

        config = load_config()
        ffprobe_path = _resolve_tool(str(_config_value(config, "ffprobe_path")), "ffprobe")
        if not ffprobe_path:
            return build_agent_response(AGENT_NAME, "failed", {}, 0.0, "ffprobe was not found for quality checks.")

        expected_duration = input_data.get("expected_duration")
        expected_duration = float(expected_duration) if expected_duration is not None else None
        actual_duration = _probe_duration(rendered_clip_path, ffprobe_path)

        if expected_duration is None:
            duration_status = "pass" if actual_duration > 0.0 else "fail"
        else:
            duration_status = "pass" if abs(actual_duration - expected_duration) <= 1.0 else "fail"

        black_details = _black_frame_check(
            rendered_clip_path,
            sample_count=int(_config_value(config, "quality_frame_sample_count")),
            black_threshold=float(_config_value(config, "quality_black_frame_threshold")),
            fail_ratio=float(_config_value(config, "quality_black_frame_fail_ratio")),
        )
        audio_details = _audio_check(
            rendered_clip_path,
            energy_threshold=float(_config_value(config, "quality_audio_energy_threshold")),
        )

        checks = {
            "black_frames": black_details["status"],
            "duration_check": duration_status,
            "audio_check": audio_details["status"],
        }
        overall = "pass" if all(value == "pass" for value in checks.values()) else "fail"

        data = {
            **checks,
            "overall": overall,
            "details": {
                "actual_duration": float(actual_duration),
                "expected_duration": expected_duration,
                "black_frame_details": black_details,
                "audio_details": audio_details,
            },
        }
        logger.info("Quality check completed: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success" if overall == "pass" else "partial",
            data=data,
            confidence=1.0 if overall == "pass" else 0.7,
            error=None if overall == "pass" else "One or more quality checks failed.",
        )

    except Exception as exc:
        logger.exception("Quality Agent failed.")
        return build_agent_response(AGENT_NAME, "failed", {}, 0.0, str(exc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Quality Agent.")
    parser.add_argument("--rendered-clip-path", required=True)
    parser.add_argument("--expected-duration", type=float, default=None)
    args = parser.parse_args()

    print(
        json.dumps(
            check_quality(
                {
                    "rendered_clip_path": args.rendered_clip_path,
                    "expected_duration": args.expected_duration,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
