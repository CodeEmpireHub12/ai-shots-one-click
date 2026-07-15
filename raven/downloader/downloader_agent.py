"""Downloader Agent for Project Raven.

Uses yt-dlp to download a public YouTube video locally.
Returns the standard Project Raven agent response envelope.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "downloader_agent"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = PROJECT_ROOT / "projects"


logger = get_agent_logger(AGENT_NAME)


def _get_ytdlp_command() -> Optional[list[str]]:
    """Return an available yt-dlp command, or None if yt-dlp is missing."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]

    python_executable = shutil.which("python") or shutil.which("python3")
    if python_executable:
        return [python_executable, "-m", "yt_dlp"]

    return None


def _run_command(command: list[str]) -> Tuple[int, str, str]:
    """Run a command and return returncode, stdout, stderr."""
    logger.info("Running command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout, completed.stderr


def _load_video_info(ytdlp_command: list[str], youtube_url: str) -> Dict[str, Any]:
    """Read YouTube metadata without downloading the video."""
    command = ytdlp_command + [
        "--dump-json",
        "--no-playlist",
        youtube_url,
    ]
    returncode, stdout, stderr = _run_command(command)

    if returncode != 0:
        raise RuntimeError(stderr.strip() or "yt-dlp failed while reading video info.")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned invalid JSON metadata.") from exc


def _pick_caption_language(info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Check captions/auto-captions and choose one language to download."""
    subtitles = info.get("subtitles") or {}
    automatic_captions = info.get("automatic_captions") or {}

    available_languages = list(subtitles.keys()) + list(automatic_captions.keys())
    if not available_languages:
        return False, None

    preferred_languages = ["en", "en-US", "en-GB"]
    for language in preferred_languages:
        if language in available_languages:
            return True, language

    return True, available_languages[0]


def _find_downloaded_video(project_dir: Path) -> Optional[Path]:
    """Find the normalized downloaded source video inside the project folder."""
    source_mp4 = project_dir / "source.mp4"
    return source_mp4 if source_mp4.exists() else None


def _clear_previous_media_files(project_dir: Path) -> None:
    """Remove previous media files before a retry, but keep captions/logs."""
    for extension in ("*.mp4", "*.mkv", "*.webm", "*.mov"):
        for path in project_dir.glob(extension):
            path.unlink(missing_ok=True)


def _finalize_downloaded_video(project_dir: Path) -> Optional[Path]:
    """
    Normalize the downloaded file to source.mp4.

    If yt-dlp leaves separate video/audio files because FFmpeg is missing,
    this returns None so the retry path can use a safer progressive format.
    """
    source_mp4 = project_dir / "source.mp4"
    if source_mp4.exists():
        return source_mp4

    media_files = []
    for extension in ("*.mp4", "*.mkv", "*.webm", "*.mov"):
        media_files.extend(project_dir.glob(extension))

    if len(media_files) != 1:
        logger.warning("Download did not produce one merged media file: %s", media_files)
        return None

    downloaded_file = media_files[0]
    downloaded_file.rename(source_mp4)
    return source_mp4


def _find_caption_file(project_dir: Path) -> Optional[Path]:
    """Find a downloaded caption file inside the project folder."""
    for extension in ("*.vtt", "*.srt", "*.ass", "*.ttml"):
        matches = list(project_dir.glob(extension))
        if matches:
            return matches[0]
    return None


def _download_with_format(
    ytdlp_command: list[str],
    youtube_url: str,
    project_dir: Path,
    format_flag: str,
    caption_language: Optional[str],
) -> None:
    """Download video/audio using a specific yt-dlp format flag."""
    command = ytdlp_command + [
        "--no-playlist",
        "--format",
        format_flag,
        "--merge-output-format",
        "mp4",
        "--output",
        str(project_dir / "source.%(ext)s"),
    ]

    if caption_language:
        command.extend([
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            caption_language,
            "--sub-format",
            "vtt/best",
        ])

    command.append(youtube_url)

    returncode, stdout, stderr = _run_command(command)
    if returncode != 0:
        raise RuntimeError(stderr.strip() or stdout.strip() or "yt-dlp download failed.")


def download_youtube_video(
    youtube_url: str,
    project_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Download a YouTube video with yt-dlp.

    Required input is youtube_url. project_dir is optional so the Input
    Normalizer can direct downloads into projects/<project_id>/.
    """
    try:
        if not youtube_url or not isinstance(youtube_url, str):
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="youtube_url must be a non-empty string.",
            )

        ytdlp_command = _get_ytdlp_command()
        if not ytdlp_command:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="yt-dlp is not installed or not available on PATH.",
            )

        target_dir = Path(project_dir) if project_dir else PROJECTS_DIR / "downloads"
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting YouTube download for URL: %s", youtube_url)

        info = _load_video_info(ytdlp_command, youtube_url)
        captions_available, caption_language = _pick_caption_language(info)

        errors = []
        format_attempts = ["bv*+ba/b", "best"]

        video_path = None
        for index, format_flag in enumerate(format_attempts, start=1):
            try:
                logger.info(
                    "Download attempt %s using format flag: %s",
                    index,
                    format_flag,
                )
                _clear_previous_media_files(target_dir)
                _download_with_format(
                    ytdlp_command=ytdlp_command,
                    youtube_url=youtube_url,
                    project_dir=target_dir,
                    format_flag=format_flag,
                    caption_language=caption_language if captions_available else None,
                )
                video_path = _finalize_downloaded_video(target_dir)
                if not video_path:
                    raise RuntimeError(
                        "yt-dlp did not produce a single merged source.mp4 file. "
                        "Retrying with fallback format."
                    )
                break
            except Exception as exc:
                error_message = str(exc)
                logger.error("Download attempt %s failed: %s", index, error_message)
                errors.append(error_message)
                if index == len(format_attempts):
                    return build_agent_response(
                        agent=AGENT_NAME,
                        status="failed",
                        data={},
                        confidence=0.0,
                        error="Download failed after retry. Last error: " + error_message,
                    )

        if not video_path:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="yt-dlp completed but no downloaded video file was found.",
            )

        caption_path = _find_caption_file(target_dir)

        width = info.get("width")
        height = info.get("height")
        resolution = info.get("resolution") or (f"{width}x{height}" if width and height else "")

        data = {
            "video_path": str(video_path),
            "title": info.get("title"),
            "description": info.get("description"),
            "duration": float(info.get("duration") or 0.0),
            "captions_available": bool(captions_available),
            "captions_path": str(caption_path) if caption_path else None,
            "resolution": resolution,
        }

        logger.info("YouTube download completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Unexpected downloader failure.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(download_youtube_video(url), indent=2))
