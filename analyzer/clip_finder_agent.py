"""Clip Finder Agent for Project Raven.

Builds candidate clip windows by combining scene motion, speech energy,
and keyword/hook/question signals. This agent only returns JSON decisions;
it does not touch media files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "clip_finder_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "clip_window_min_seconds": 20.0,
    "clip_window_max_seconds": 60.0,
    "clip_window_step_seconds": 5.0,
    "clip_top_candidates": 20,
    "clip_motion_weight": 1.0,
    "clip_energy_weight": 1.0,
    "clip_hook_weight": 2.0,
    "clip_question_weight": 1.5,
    "clip_keyword_weight": 1.0,
}


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return overlap duration between two time ranges."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _validate_range_items(items: Any, required_fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Validate list items with start/end float timestamps."""
    if not isinstance(items, list):
        return []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except Exception:
            continue
        if end <= start:
            continue

        normalized_item = dict(item)
        normalized_item["start"] = start
        normalized_item["end"] = end

        missing_required = any(field not in normalized_item for field in required_fields)
        if missing_required:
            continue

        normalized.append(normalized_item)

    normalized.sort(key=lambda entry: entry["start"])
    return normalized


def _video_duration(
    scenes: List[Dict[str, Any]],
    segment_features: List[Dict[str, Any]],
    hooks: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    keywords: List[Dict[str, Any]],
) -> float:
    """Infer video duration from all available timestamped inputs."""
    ends = []
    for collection in (scenes, segment_features, hooks, questions, keywords):
        ends.extend(float(item["end"]) for item in collection if "end" in item)
    return max(ends) if ends else 0.0


def _build_windows(duration: float, min_seconds: float, max_seconds: float, step_seconds: float) -> List[Dict[str, float]]:
    """Build sliding candidate windows."""
    if duration <= 0.0:
        return []

    safe_min = max(1.0, float(min_seconds))
    safe_max = max(safe_min, float(max_seconds))
    safe_step = max(1.0, float(step_seconds))

    if duration <= safe_min:
        return [{"start": 0.0, "end": float(duration)}]

    window_lengths = sorted(set([safe_min, min(safe_max, duration)]))
    windows = []
    seen = set()

    for window_length in window_lengths:
        start = 0.0
        while start < duration:
            end = min(duration, start + window_length)
            if end - start >= min(safe_min, duration):
                key = (round(start, 3), round(end, 3))
                if key not in seen:
                    windows.append({"start": float(start), "end": float(end)})
                    seen.add(key)
            if end >= duration:
                break
            start += safe_step

    return windows


def _average_scene_motion(window: Dict[str, float], scenes: List[Dict[str, Any]]) -> float:
    """Weighted average scene motion inside a window."""
    weighted_total = 0.0
    overlap_total = 0.0
    for scene in scenes:
        overlap = _overlap_seconds(window["start"], window["end"], scene["start"], scene["end"])
        if overlap <= 0.0:
            continue
        weighted_total += overlap * float(scene.get("motion_score", 0.0))
        overlap_total += overlap
    return float(weighted_total / overlap_total) if overlap_total > 0.0 else 0.0


def _average_speech_energy(window: Dict[str, float], segment_features: List[Dict[str, Any]]) -> float:
    """Weighted average speech energy inside a window."""
    weighted_total = 0.0
    overlap_total = 0.0
    for segment in segment_features:
        overlap = _overlap_seconds(window["start"], window["end"], segment["start"], segment["end"])
        if overlap <= 0.0:
            continue
        weighted_total += overlap * float(segment.get("energy", 0.0))
        overlap_total += overlap
    return float(weighted_total / overlap_total) if overlap_total > 0.0 else 0.0


def _count_overlapping(window: Dict[str, float], items: List[Dict[str, Any]]) -> int:
    """Count timestamped items that overlap a window."""
    count = 0
    for item in items:
        if _overlap_seconds(window["start"], window["end"], item["start"], item["end"]) > 0.0:
            count += 1
    return count


def _keyword_score(window: Dict[str, float], keywords: List[Dict[str, Any]]) -> float:
    """Sum keyword scores for keywords overlapping the window."""
    score = 0.0
    for keyword in keywords:
        if _overlap_seconds(window["start"], window["end"], keyword["start"], keyword["end"]) > 0.0:
            score += float(keyword.get("score", 1.0))
    return float(score)


def _score_window(
    window: Dict[str, float],
    scenes: List[Dict[str, Any]],
    segment_features: List[Dict[str, Any]],
    hooks: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    keywords: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Score one candidate window."""
    motion = _average_scene_motion(window, scenes)
    energy = _average_speech_energy(window, segment_features)
    hook_count = _count_overlapping(window, hooks)
    question_count = _count_overlapping(window, questions)
    keyword_total = _keyword_score(window, keywords)

    raw_score = (
        motion * weights["motion"]
        + energy * weights["energy"]
        + hook_count * weights["hook"]
        + question_count * weights["question"]
        + keyword_total * weights["keyword"]
    )

    return {
        "start": float(window["start"]),
        "end": float(window["end"]),
        "raw_score": float(raw_score),
        "signals": {
            "motion": float(motion),
            "energy": float(energy),
            "hook_count": int(hook_count),
            "question_count": int(question_count),
            "keyword_score": float(keyword_total),
        },
    }


def find_candidate_clips(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rank sliding windows as candidate clips.

    Expected input:
    {
      "scenes": [{"start": 0.0, "end": 10.0, "motion_score": 0.1, "brightness": 0.5}],
      "segment_features": [{"start": 0.0, "end": 2.0, "energy": 0.1, "pace": 3.0, "pause_before": 0.0}],
      "hooks": [{"start": 0.0, "end": 2.0, "text": "..."}],
      "questions": [{"start": 3.0, "end": 5.0, "text": "..."}],
      "keywords": [{"text": "example", "score": 1.0, "start": 0.0, "end": 2.0}]
    }
    """
    try:
        scenes = _validate_range_items(input_data.get("scenes"), ("motion_score", "brightness"))
        segment_features = _validate_range_items(input_data.get("segment_features"), ("energy", "pace", "pause_before"))
        hooks = _validate_range_items(input_data.get("hooks"), ("text",))
        questions = _validate_range_items(input_data.get("questions"), ("text",))
        keywords = _validate_range_items(input_data.get("keywords"), ("text", "score"))

        if not scenes and not segment_features:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="At least scenes or segment_features are required.",
            )

        config = load_config()
        min_seconds = float(_config_value(config, "clip_window_min_seconds"))
        max_seconds = float(_config_value(config, "clip_window_max_seconds"))
        step_seconds = float(_config_value(config, "clip_window_step_seconds"))
        top_candidates = int(_config_value(config, "clip_top_candidates"))
        weights = {
            "motion": float(_config_value(config, "clip_motion_weight")),
            "energy": float(_config_value(config, "clip_energy_weight")),
            "hook": float(_config_value(config, "clip_hook_weight")),
            "question": float(_config_value(config, "clip_question_weight")),
            "keyword": float(_config_value(config, "clip_keyword_weight")),
        }

        duration = _video_duration(scenes, segment_features, hooks, questions, keywords)
        windows = _build_windows(duration, min_seconds, max_seconds, step_seconds)
        if not windows:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="Could not build candidate windows from provided timestamps.",
            )

        scored_windows = [
            _score_window(window, scenes, segment_features, hooks, questions, keywords, weights)
            for window in windows
        ]
        candidates = sorted(scored_windows, key=lambda item: item["raw_score"], reverse=True)[:top_candidates]

        data = {"candidates": candidates}
        logger.info("Clip finding completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Clip Finder Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Clip Finder Agent.")
    parser.add_argument("--scene-json", required=True)
    parser.add_argument("--speech-json", required=True)
    parser.add_argument("--keyword-json", required=True)
    args = parser.parse_args()

    with Path(args.scene_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        scene_response = json.load(file)
    with Path(args.speech_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        speech_response = json.load(file)
    with Path(args.keyword_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        keyword_response = json.load(file)

    scene_data = scene_response.get("data", scene_response)
    speech_data = speech_response.get("data", speech_response)
    keyword_data = keyword_response.get("data", keyword_response)

    print(
        json.dumps(
            find_candidate_clips(
                {
                    "scenes": scene_data.get("scenes", []),
                    "segment_features": speech_data.get("segment_features", []),
                    "hooks": keyword_data.get("hooks", []),
                    "questions": keyword_data.get("questions", []),
                    "keywords": keyword_data.get("keywords", []),
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
