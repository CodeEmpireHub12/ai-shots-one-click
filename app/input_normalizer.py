"""Input Normalizer for Project Raven.

Accepts either a YouTube URL or a local upload path and converts it into
Project Raven's unified input structure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response
from raven.downloader.downloader_agent import download_youtube_video


AGENT_NAME = "input_normalizer"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = PROJECT_ROOT / "projects"
SUPPORTED_UPLOAD_FORMATS = {".mp4", ".mov", ".mkv"}


logger = get_agent_logger(AGENT_NAME)


def _create_project_id() -> str:
    """Create a Raven project id using a millisecond Unix timestamp."""
    return f"raven_{int(time.time() * 1000)}"


def _empty_normalized_data(project_id: str, source_type: str) -> Dict[str, Any]:
    """Create the unified normalized data object with default values."""
    return {
        "project_id": project_id,
        "video_path": "",
        "duration": 0.0,
        "resolution": "",
        "source_type": source_type,
        "youtube_captions_available": False,
        "original_title": None,
    }


def _save_normalized_input(
    project_dir: Path,
    normalized_data: Dict[str, Any],
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Save project-side normalized input without changing agent output schema."""
    project_file_data = dict(normalized_data)
    if extra_data:
        project_file_data.update(extra_data)

    project_dir.mkdir(parents=True, exist_ok=True)
    with (project_dir / "normalized_input.json").open("w", encoding="utf-8") as file:
        json.dump(project_file_data, file, indent=2, ensure_ascii=False)


def _run_ffprobe(video_path: Path) -> Tuple[float, str]:
    """
    Read duration and resolution using ffprobe if available.

    If ffprobe is missing or fails, keep safe default values. This is
    deterministic metadata reading, not AI media processing.
    """
    ffprobe_command = shutil.which("ffprobe")
    if not ffprobe_command:
        logger.warning("ffprobe not found. Duration and resolution will use defaults.")
        return 0.0, ""

    command = [
        ffprobe_command,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
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
        logger.warning("ffprobe failed: %s", completed.stderr.strip())
        return 0.0, ""

    try:
        probe_data = json.loads(completed.stdout)
        duration = float((probe_data.get("format") or {}).get("duration") or 0.0)
        stream = (probe_data.get("streams") or [{}])[0]
        width = stream.get("width")
        height = stream.get("height")
        resolution = f"{width}x{height}" if width and height else ""
        return duration, resolution
    except Exception as exc:
        logger.warning("Could not parse ffprobe output: %s", exc)
        return 0.0, ""


def _normalize_upload(file_path: str, project_id: str, project_dir: Path) -> Dict[str, Any]:
    """Validate and copy a local upload into the Raven project folder."""
    source_path = Path(file_path).expanduser().resolve()

    if not source_path.exists():
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=f"Upload file does not exist: {source_path}",
        )

    if not source_path.is_file():
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=f"Upload path is not a file: {source_path}",
        )

    if source_path.suffix.lower() not in SUPPORTED_UPLOAD_FORMATS:
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error="Unsupported upload format. Supported formats: mp4, mov, mkv.",
        )

    project_dir.mkdir(parents=True, exist_ok=True)
    target_path = project_dir / "source.mp4"
    shutil.copy2(source_path, target_path)

    duration, resolution = _run_ffprobe(target_path)

    data = _empty_normalized_data(project_id=project_id, source_type="upload")
    data.update(
        {
            "video_path": str(target_path),
            "duration": duration,
            "resolution": resolution,
            "youtube_captions_available": False,
            "original_title": source_path.stem,
        }
    )

    _save_normalized_input(project_dir=project_dir, normalized_data=data)

    logger.info("Upload input normalized successfully: %s", data)
    return build_agent_response(
        agent=AGENT_NAME,
        status="success",
        data=data,
        confidence=1.0,
        error=None,
    )


def _normalize_youtube(youtube_url: str, project_id: str, project_dir: Path) -> Dict[str, Any]:
    """Call the Downloader Agent and convert its result into unified input data."""
    downloader_response = download_youtube_video(
        youtube_url=youtube_url,
        project_dir=project_dir,
    )

    if downloader_response.get("status") == "failed":
        logger.error("Downloader Agent failed: %s", downloader_response.get("error"))
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=downloader_response.get("error") or "Downloader Agent failed.",
        )

    downloader_data = downloader_response.get("data") or {}
    video_path = Path(downloader_data.get("video_path", ""))

    probed_duration, probed_resolution = _run_ffprobe(video_path) if video_path.exists() else (0.0, "")
    duration = float(downloader_data.get("duration") or probed_duration or 0.0)
    resolution = probed_resolution or str(downloader_data.get("resolution") or "")

    data = _empty_normalized_data(project_id=project_id, source_type="youtube")
    data.update(
        {
            "video_path": str(video_path),
            "duration": duration,
            "resolution": resolution,
            "youtube_captions_available": bool(downloader_data.get("captions_available", False)),
            "original_title": downloader_data.get("title"),
        }
    )

    _save_normalized_input(
        project_dir=project_dir,
        normalized_data=data,
        extra_data={
            "downloader": {
                "captions_path": downloader_data.get("captions_path"),
                "description": downloader_data.get("description"),
            }
        },
    )

    status = "success" if data["video_path"] and data["duration"] > 0 else "partial"
    confidence = 1.0 if status == "success" else 0.6

    logger.info("YouTube input normalized with status %s: %s", status, data)
    return build_agent_response(
        agent=AGENT_NAME,
        status=status,
        data=data,
        confidence=confidence,
        error=None if status == "success" else "Normalized, but some metadata is missing.",
    )


def normalize_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a YouTube URL or local upload into one Raven input structure.

    Expected input examples:
    {"youtube_url": "https://www.youtube.com/watch?v=..."}
    {"file_path": "/path/to/local/video.mp4"}
    """
    try:
        youtube_url = input_data.get("youtube_url")
        file_path = input_data.get("file_path")

        if bool(youtube_url) == bool(file_path):
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="Provide exactly one input: youtube_url or file_path.",
            )

        project_id = _create_project_id()
        project_dir = PROJECTS_DIR / project_id

        if youtube_url:
            logger.info("Normalizing YouTube input for project_id=%s", project_id)
            return _normalize_youtube(
                youtube_url=str(youtube_url),
                project_id=project_id,
                project_dir=project_dir,
            )

        logger.info("Normalizing upload input for project_id=%s", project_id)
        return _normalize_upload(
            file_path=str(file_path),
            project_id=project_id,
            project_dir=project_dir,
        )

    except Exception as exc:
        logger.exception("Unexpected input normalization failure.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3 or sys.argv[1] not in {"--youtube-url", "--file-path"}:
        print("Usage: python -m raven.app.input_normalizer --youtube-url <url>")
        print("   or: python -m raven.app.input_normalizer --file-path <path>")
        raise SystemExit(1)

    key = "youtube_url" if sys.argv[1] == "--youtube-url" else "file_path"
    print(json.dumps(normalize_input({key: sys.argv[2]}), indent=2))
