"""Caption Agent for Project Raven.

Generates a simple ASS subtitle file for one selected clip. The agent
extracts transcript segments overlapping the clip and rewrites subtitle
timestamps relative to the clip start.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "caption_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "caption_font_name": "Arial",
    "caption_font_size": 54,
    "caption_margin_v": 220,
}


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _ass_time(seconds: float) -> str:
    """Convert float seconds to ASS timestamp format H:MM:SS.cs."""
    safe_seconds = max(0.0, float(seconds))
    hours = int(safe_seconds // 3600)
    minutes = int((safe_seconds % 3600) // 60)
    whole_seconds = int(safe_seconds % 60)
    centiseconds = int(round((safe_seconds - int(safe_seconds)) * 100))

    if centiseconds >= 100:
        whole_seconds += 1
        centiseconds -= 100
    if whole_seconds >= 60:
        minutes += 1
        whole_seconds -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60

    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape text for ASS dialogue lines."""
    clean_text = str(text).replace("\n", " ").replace("\r", " ").strip()
    clean_text = clean_text.replace("{", "(").replace("}", ")")
    return " ".join(clean_text.split())


def _validate_clip(clip: Any) -> Dict[str, float]:
    """Validate one selected clip object."""
    if not isinstance(clip, dict):
        raise ValueError("clip must be an object with start and end.")

    start = float(clip.get("start", 0.0))
    end = float(clip.get("end", 0.0))
    if end <= start:
        raise ValueError("clip end must be greater than clip start.")

    return {"start": start, "end": end}


def _validate_segments(segments: Any) -> List[Dict[str, Any]]:
    """Validate transcript segments."""
    if not isinstance(segments, list):
        raise ValueError("segments must be a list.")

    normalized = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text = str(segment.get("text", "")).strip()
        if end <= start or not text:
            continue
        normalized.append({"start": start, "end": end, "text": text})

    normalized.sort(key=lambda item: item["start"])
    return normalized


def _extract_clip_segments(
    clip: Dict[str, float],
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract overlapping transcript segments and make timestamps clip-relative."""
    clip_start = float(clip["start"])
    clip_end = float(clip["end"])
    clip_segments = []

    for segment in segments:
        overlap_start = max(float(segment["start"]), clip_start)
        overlap_end = min(float(segment["end"]), clip_end)
        if overlap_end <= overlap_start:
            continue

        relative_start = overlap_start - clip_start
        relative_end = overlap_end - clip_start
        clip_segments.append(
            {
                "start": float(relative_start),
                "end": float(relative_end),
                "text": segment["text"],
            }
        )

    return clip_segments


def _build_ass_content(
    clip_segments: List[Dict[str, Any]],
    font_name: str,
    font_size: int,
    margin_v: int,
) -> str:
    """Build ASS subtitle file content."""
    header = f"""[Script Info]
Title: Project Raven Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines = []
    for segment in clip_segments:
        start = _ass_time(segment["start"])
        end = _ass_time(segment["end"])
        text = _escape_ass_text(segment["text"])
        dialogue_lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    return header + "\n".join(dialogue_lines) + "\n"


def generate_captions(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate captions.ass for one selected clip.

    Expected input:
    {
      "clip": {"start": 0.0, "end": 20.0},
      "segments": [{"start": 1.0, "end": 2.0, "text": "..."}],
      "project_dir": "/path/to/raven/projects/<project_id>",
      "clip_index": 1
    }
    """
    try:
        clip = _validate_clip(input_data.get("clip"))
        segments = _validate_segments(input_data.get("segments"))
        if not segments:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="No valid transcript segments were provided.",
            )

        project_dir_value = input_data.get("project_dir")
        if not project_dir_value:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="project_dir is required.",
            )

        project_dir = Path(str(project_dir_value)).expanduser().resolve()
        clip_index = int(input_data.get("clip_index", 1))
        if clip_index < 1:
            clip_index = 1

        clip_segments = _extract_clip_segments(clip, segments)
        if not clip_segments:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="No transcript segments overlap the selected clip.",
            )

        config = load_config()
        font_name = str(_config_value(config, "caption_font_name"))
        font_size = int(_config_value(config, "caption_font_size"))
        margin_v = int(_config_value(config, "caption_margin_v"))

        clip_dir = project_dir / "clips" / f"clip_{clip_index}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        ass_path = clip_dir / "captions.ass"

        ass_content = _build_ass_content(
            clip_segments=clip_segments,
            font_name=font_name,
            font_size=font_size,
            margin_v=margin_v,
        )
        ass_path.write_text(ass_content, encoding="utf-8")

        data = {
            "ass_path": str(ass_path),
            "caption_segments": clip_segments,
        }
        logger.info("Caption generation completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Caption Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Caption Agent.")
    parser.add_argument("--selected-clips-json", required=True)
    parser.add_argument("--transcript-json", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--clip-index", type=int, default=1)
    args = parser.parse_args()

    with Path(args.selected_clips_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        selected_response = json.load(file)
    with Path(args.transcript_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        transcript_data = json.load(file)

    selected_data = selected_response.get("data", selected_response)
    clips = selected_data.get("selected_clips", [])
    if not clips:
        print(json.dumps(build_agent_response(AGENT_NAME, "failed", {}, 0.0, "No selected clips found."), indent=2))
        raise SystemExit(0)

    print(
        json.dumps(
            generate_captions(
                {
                    "clip": clips[args.clip_index - 1],
                    "segments": transcript_data.get("segments", []),
                    "project_dir": args.project_dir,
                    "clip_index": args.clip_index,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
