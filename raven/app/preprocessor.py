"""Preprocessor for Project Raven.

Extracts Whisper-compatible audio and reads deterministic video metadata
with ffmpeg/ffprobe.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "preprocessor"
logger = get_agent_logger(AGENT_NAME)


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    logger.info("Running command: %s", " ".join(command))
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _resolve_tool(config_value: Optional[str], fallback_name: str) -> Optional[str]:
    """Resolve a tool path from config or PATH."""
    if config_value:
        configured_path = Path(str(config_value)).expanduser()
        if configured_path.exists():
            return str(configured_path)
        found_config_name = shutil.which(str(config_value))
        if found_config_name:
            return found_config_name

    return shutil.which(fallback_name)


def _get_metadata(video_path: Path, ffprobe_path: str) -> Dict[str, Any]:
    """Get duration, resolution, and fps from ffprobe."""
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = _run_command(command)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe failed.")

    probe_data = json.loads(completed.stdout)
    stream = (probe_data.get("streams") or [{}])[0]
    width = stream.get("width")
    height = stream.get("height")
    resolution = f"{width}x{height}" if width and height else ""

    fps = 0.0
    frame_rate = stream.get("r_frame_rate") or "0/1"
    if "/" in frame_rate:
        numerator, denominator = frame_rate.split("/", 1)
        denominator_float = float(denominator or 1)
        fps = float(numerator or 0) / denominator_float if denominator_float else 0.0
    else:
        fps = float(frame_rate or 0.0)

    duration = float((probe_data.get("format") or {}).get("duration") or 0.0)

    return {
        "duration": duration,
        "resolution": resolution,
        "fps": fps,
    }


def _extract_audio(video_path: Path, audio_path: Path, ffmpeg_path: str) -> None:
    """Extract 16 kHz mono WAV audio for Whisper compatibility."""
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    completed = _run_command(command)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg audio extraction failed.")


def _load_project_input(project_dir: Path) -> Dict[str, Any]:
    """Load existing normalized project input if present."""
    input_path = project_dir / "normalized_input.json"
    if not input_path.exists():
        return {}

    try:
        with input_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        logger.warning("Could not read normalized_input.json: %s", exc)
        return {}


def _save_project_input(project_dir: Path, data: Dict[str, Any]) -> Path:
    """Save updated project-side input/metadata JSON."""
    input_path = project_dir / "normalized_input.json"
    with input_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return input_path


def preprocess_video(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a video path.

    Expected input:
    {"video_path": "/path/to/project/source.mp4"}
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
        ffmpeg_path = _resolve_tool(config.get("ffmpeg_path"), "ffmpeg")
        ffprobe_path = _resolve_tool(config.get("ffprobe_path"), "ffprobe")

        if not ffmpeg_path:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="ffmpeg was not found. Set ffmpeg_path in settings/config.yaml or add ffmpeg to PATH.",
            )

        if not ffprobe_path:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="ffprobe was not found. Install FFmpeg tools or add ffprobe to PATH.",
            )

        project_dir = video_path.parent
        audio_path = project_dir / "audio.wav"

        metadata = _get_metadata(video_path, ffprobe_path)
        _extract_audio(video_path, audio_path, ffmpeg_path)

        project_input = _load_project_input(project_dir)
        project_input.update(
            {
                "video_path": str(video_path),
                "duration": float(metadata["duration"]),
                "resolution": metadata["resolution"],
                "preprocessor": {
                    "audio_path": str(audio_path),
                    "fps": float(metadata["fps"]),
                },
            }
        )
        normalized_input_path = _save_project_input(project_dir, project_input)

        data = {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "duration": float(metadata["duration"]),
            "resolution": metadata["resolution"],
            "fps": float(metadata["fps"]),
            "normalized_input_path": str(normalized_input_path),
        }

        logger.info("Preprocessing completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Preprocessing failed.")
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
        print("Usage: python -m raven.app.preprocessor <video_path>")
        raise SystemExit(1)

    print(json.dumps(preprocess_video({"video_path": sys.argv[1]}), indent=2))
