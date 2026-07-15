"""Export Agent for Project Raven.

Executes a Timeline Agent FFmpeg command, verifies the rendered file
exists, and checks that it has non-zero duration.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "export_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "ffprobe_path": "ffprobe",
}


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _extract_output_path(ffmpeg_command: str, explicit_output_path: Optional[str]) -> Path:
    """Get output path from explicit input or final FFmpeg command token."""
    if explicit_output_path:
        return Path(str(explicit_output_path)).expanduser().resolve()

    parts = shlex.split(ffmpeg_command)
    if not parts:
        raise ValueError("ffmpeg_command is empty.")
    return Path(parts[-1]).expanduser().resolve()


def _resolve_tool(config_value: str, fallback_name: str) -> Optional[str]:
    """Resolve ffprobe from config or PATH."""
    configured = Path(str(config_value)).expanduser()
    if configured.exists():
        return str(configured)

    from_config_name = shutil.which(str(config_value))
    if from_config_name:
        return from_config_name

    return shutil.which(fallback_name)


def _probe_duration(output_path: Path, ffprobe_path: str) -> float:
    """Return output duration from ffprobe."""
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe failed while verifying output.")

    probe_data = json.loads(completed.stdout)
    return float((probe_data.get("format") or {}).get("duration") or 0.0)


def export_video(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute FFmpeg command from Timeline Agent.

    Expected input:
    {
      "ffmpeg_command": "ffmpeg ... final.mp4",
      "output_path": "/path/to/final.mp4"  # optional but preferred
    }
    """
    try:
        ffmpeg_command = str(input_data.get("ffmpeg_command", "")).strip()
        if not ffmpeg_command:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="ffmpeg_command is required.",
            )

        output_path = _extract_output_path(ffmpeg_command, input_data.get("output_path"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Executing FFmpeg command: %s", ffmpeg_command)
        start_time = time.time()
        completed = subprocess.run(
            ffmpeg_command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        render_time_sec = float(time.time() - start_time)

        if completed.returncode != 0:
            logger.error("FFmpeg failed: %s", completed.stderr)
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={
                    "output_path": str(output_path),
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                confidence=0.0,
                error=completed.stderr.strip() or "FFmpeg failed with a non-zero exit code.",
            )

        if not output_path.exists() or not output_path.is_file():
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={"stdout": completed.stdout, "stderr": completed.stderr},
                confidence=0.0,
                error=f"FFmpeg completed but output file was not created: {output_path}",
            )

        file_size_mb = float(output_path.stat().st_size / (1024 * 1024))
        if file_size_mb <= 0.0:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={"output_path": str(output_path), "file_size_mb": file_size_mb},
                confidence=0.0,
                error="Output file was created but has zero size.",
            )

        config = load_config()
        ffprobe_path = _resolve_tool(str(_config_value(config, "ffprobe_path")), "ffprobe")
        if not ffprobe_path:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={"output_path": str(output_path), "file_size_mb": file_size_mb},
                confidence=0.0,
                error="ffprobe was not found, so output duration could not be verified.",
            )

        duration = _probe_duration(output_path, ffprobe_path)
        if duration <= 0.0:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={"output_path": str(output_path), "file_size_mb": file_size_mb, "duration": duration},
                confidence=0.0,
                error="Output file duration is zero.",
            )

        data = {
            "output_path": str(output_path),
            "file_size_mb": file_size_mb,
            "render_time_sec": render_time_sec,
        }
        logger.info("Export completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Export Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Export Agent.")
    parser.add_argument("--timeline-json", required=True)
    args = parser.parse_args()

    with Path(args.timeline_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        timeline_response = json.load(file)

    timeline_data = timeline_response.get("data", timeline_response)
    print(
        json.dumps(
            export_video(
                {
                    "ffmpeg_command": timeline_data.get("ffmpeg_command", ""),
                    "output_path": timeline_data.get("output_path"),
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
