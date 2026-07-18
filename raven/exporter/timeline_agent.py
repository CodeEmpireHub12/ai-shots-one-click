"""Timeline Agent for Project Raven.

Builds deterministic FFmpeg command instructions for rendering one
vertical short with burned-in ASS captions. This agent only builds JSON
instructions; it does not execute FFmpeg.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, Tuple

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "timeline_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "ffmpeg_path": "ffmpeg",
    "output_resolution": "1080x1920",
    "export_video_codec": "libx264",
    "export_audio_codec": "aac",
    "export_preset": "veryfast",
    "export_crf": 23,
    "export_audio_bitrate": "128k",
}


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _parse_resolution(value: str) -> Tuple[int, int]:
    """Parse resolution string like 1080x1920."""
    parts = str(value).lower().split("x")
    if len(parts) != 2:
        raise ValueError("output_resolution must look like 1080x1920.")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("output_resolution width/height must be positive.")
    return width, height


def _validate_clip(clip: Any) -> Dict[str, Any]:
    """Validate clip data and normalize numeric fields."""
    if not isinstance(clip, dict):
        raise ValueError("clip must be an object.")

    start = float(clip.get("start", 0.0))
    end = float(clip.get("end", 0.0))
    video_path_value = clip.get("video_path")

    if end <= start:
        raise ValueError("clip end must be greater than clip start.")
    if not video_path_value:
        raise ValueError("clip.video_path is required.")

    video_path = Path(str(video_path_value)).expanduser().resolve()
    if not video_path.exists() or not video_path.is_file():
        raise ValueError(f"Video file does not exist: {video_path}")

    return {
        "start": start,
        "end": end,
        "video_path": video_path,
    }


def _escape_subtitles_filter_path(path: Path) -> str:
    """Escape a path for FFmpeg subtitles filter syntax."""
    text = str(path)
    # Convert Windows backslashes to forward slashes and escape colons after drive letter
    text = text.replace("\\", "/")
    text = text.replace(":/", "\\:/")
    text = text.replace("'", "\\'")
    return text


def build_timeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build FFmpeg command/filter instructions for one clip.

    Expected input:
    {
      "clip": {"start": 0.0, "end": 20.0, "video_path": "/path/source.mp4"},
      "ass_path": "/path/projects/<id>/clips/clip_X/captions.ass",
      "clip_index": 1
    }
    """
    try:
        clip = _validate_clip(input_data.get("clip"))
        ass_path_value = input_data.get("ass_path")
        if not ass_path_value:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="ass_path is required.",
            )

        ass_path = Path(str(ass_path_value)).expanduser().resolve()
        if not ass_path.exists() or not ass_path.is_file():
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error=f"ASS subtitle file does not exist: {ass_path}",
            )

        clip_index = int(input_data.get("clip_index", 1))
        if clip_index < 1:
            clip_index = 1

        config = load_config()
        ffmpeg_path = str(_config_value(config, "ffmpeg_path"))
        output_width, output_height = _parse_resolution(str(_config_value(config, "output_resolution")))
        video_codec = str(_config_value(config, "export_video_codec"))
        audio_codec = str(_config_value(config, "export_audio_codec"))
        preset = str(_config_value(config, "export_preset"))
        crf = int(_config_value(config, "export_crf"))
        audio_bitrate = str(_config_value(config, "export_audio_bitrate"))

        project_dir = clip["video_path"].parent
        clip_dir = project_dir / "clips" / f"clip_{clip_index}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        output_path = clip_dir / "final.mp4"

        target_aspect = output_width / output_height
        escaped_ass_path = _escape_subtitles_filter_path(ass_path)
        video_filter = (
            f"scale=if(gt(a,{target_aspect}),-2,{output_width}):"
            f"if(gt(a,{target_aspect}),{output_height},-2),"
            f"crop={output_width}:{output_height}:(iw-{output_width})/2:(ih-{output_height})/2,"
            f"subtitles='{escaped_ass_path}'"
        )

        duration = clip["end"] - clip["start"]

        ffmpeg_command = (
            f"{ffmpeg_path} -y"
            f" -ss {clip['start']}"
            f" -t {duration}"
            f" -i {clip['video_path']}"
            f" -vf {video_filter}"
            f" -c:v {video_codec}"
            f" -preset {preset}"
            f" -crf {crf}"
            f" -c:a {audio_codec}"
            f" -b:a {audio_bitrate}"
            f" -movflags +faststart"
            f" {output_path}"
        )

        data = {
            "ffmpeg_command": ffmpeg_command,
            "output_path": str(output_path),
        }
        logger.info("Timeline command built successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Timeline Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Timeline Agent.")
    parser.add_argument("--selected-clips-json", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--ass-path", required=True)
    parser.add_argument("--clip-index", type=int, default=1)
    args = parser.parse_args()

    with Path(args.selected_clips_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        selected_response = json.load(file)

    selected_data = selected_response.get("data", selected_response)
    clips = selected_data.get("selected_clips", [])
    if not clips:
        print(json.dumps(build_agent_response(AGENT_NAME, "failed", {}, 0.0, "No selected clips found."), indent=2))
        raise SystemExit(0)

    selected_clip = dict(clips[args.clip_index - 1])
    selected_clip["video_path"] = args.video_path

    print(
        json.dumps(
            build_timeline(
                {
                    "clip": selected_clip,
                    "ass_path": args.ass_path,
                    "clip_index": args.clip_index,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
