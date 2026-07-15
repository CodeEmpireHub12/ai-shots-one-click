"""Director Agent for Project Raven.

Uses a local Ollama LLM to select the best candidate clips. If the LLM is
unavailable or returns invalid JSON twice, falls back to deterministic
rule-based selection by raw_score.

This agent only makes JSON decisions. It never touches media files.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "director_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "llm_model_name": "qwen3:4b",
    "ollama_host": "http://localhost:11434",
}

SYSTEM_PROMPT = """You are a professional short-form video editor.
Your task is to select the best clips for a viral vertical short.
Consider hook strength, emotional impact, story completeness, pacing, and replay value.
Return ONLY valid JSON. Do not include explanations outside JSON. Do not use markdown. Do not wrap the JSON in code fences.
The JSON must match this exact structure:
{
  "selected_clips": [
    {"start": 82.0, "end": 105.0, "score": 96, "reason": "Strong hook + emotional peak", "rank": 1}
  ]
}
"""

STRICT_RETRY_SUFFIX = """
Your previous response was not valid JSON.
Return ONLY the JSON object, wrapped in nothing else.
No markdown. No code fences. No comments. No extra text before or after the JSON.
"""

WORD_SPACE_PATTERN = re.compile(r"\s+")


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _validate_candidates(candidates: Any) -> List[Dict[str, Any]]:
    """Validate candidate clips and normalize numeric fields."""
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list.")

    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        try:
            start = float(candidate.get("start", 0.0))
            end = float(candidate.get("end", 0.0))
            raw_score = float(candidate.get("raw_score", 0.0))
        except Exception:
            continue

        if end <= start:
            continue

        normalized_candidate = dict(candidate)
        normalized_candidate["start"] = start
        normalized_candidate["end"] = end
        normalized_candidate["raw_score"] = raw_score
        normalized.append(normalized_candidate)

    normalized.sort(key=lambda item: item["raw_score"], reverse=True)
    return normalized[:20]


def _validate_segments(segments: Any) -> List[Dict[str, Any]]:
    """Validate transcript segments for context snippets."""
    if not isinstance(segments, list):
        return []

    normalized = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
        except Exception:
            continue
        text = str(segment.get("text", "")).strip()
        if end <= start or not text:
            continue
        normalized.append({"start": start, "end": end, "text": text})

    normalized.sort(key=lambda item: item["start"])
    return normalized


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return overlap duration between two timestamp ranges."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _snippet_for_candidate(candidate: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
    """Build a short transcript snippet for one candidate clip."""
    existing_snippet = str(candidate.get("transcript_snippet", "")).strip()
    if existing_snippet:
        return WORD_SPACE_PATTERN.sub(" ", existing_snippet)[:500]

    start = float(candidate["start"])
    end = float(candidate["end"])
    texts = []
    for segment in segments:
        if _overlap_seconds(start, end, segment["start"], segment["end"]) > 0.0:
            texts.append(segment["text"])

    snippet = WORD_SPACE_PATTERN.sub(" ", " ".join(texts)).strip()
    return snippet[:500]


def _build_user_prompt(candidates: List[Dict[str, Any]], segments: List[Dict[str, Any]], strict_retry: bool = False) -> str:
    """Build the user prompt sent to the local LLM."""
    prompt_candidates = []
    for index, candidate in enumerate(candidates, start=1):
        prompt_candidates.append(
            {
                "candidate_id": index,
                "start": float(candidate["start"]),
                "end": float(candidate["end"]),
                "raw_score": float(candidate["raw_score"]),
                "transcript_snippet": _snippet_for_candidate(candidate, segments),
            }
        )

    prompt = {
        "task": "Pick the best 3-5 clips for a viral short from these candidates.",
        "selection_rules": [
            "Choose clips with strong hook strength.",
            "Prefer emotional impact and story completeness.",
            "Use raw_score as one signal, not the only signal.",
            "Return 3-5 clips if enough candidates exist; otherwise return all usable candidates.",
            "Use float seconds for start and end.",
            "Score each selected clip from 0 to 100.",
        ],
        "required_output_format": {
            "selected_clips": [
                {
                    "start": 82.0,
                    "end": 105.0,
                    "score": 96,
                    "reason": "Strong hook + emotional peak",
                    "rank": 1,
                }
            ]
        },
        "candidates": prompt_candidates,
    }

    text = json.dumps(prompt, indent=2, ensure_ascii=False)
    if strict_retry:
        text = STRICT_RETRY_SUFFIX + "\n" + text
    return text


def _call_ollama(model_name: str, ollama_host: str, system_prompt: str, user_prompt: str) -> str:
    """Call local Ollama /api/generate and return raw model text."""
    host = str(ollama_host).rstrip("/")
    url = f"{host}/api/generate"
    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info("Calling Ollama model=%s host=%s", model_name, host)
    with urllib.request.urlopen(request, timeout=120) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    return str(response_data.get("response", "")).strip()


def _parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Parse and validate the LLM's required JSON output."""
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")

    selected_clips = parsed.get("selected_clips")
    if not isinstance(selected_clips, list) or not selected_clips:
        raise ValueError("LLM response must include non-empty selected_clips list.")

    normalized_selected = []
    for index, clip in enumerate(selected_clips, start=1):
        if not isinstance(clip, dict):
            continue

        start = float(clip.get("start", 0.0))
        end = float(clip.get("end", 0.0))
        score = int(round(float(clip.get("score", 0))))
        reason = str(clip.get("reason", "")).strip() or "Selected by local LLM"
        rank = int(clip.get("rank", index))

        if end <= start:
            continue

        normalized_selected.append(
            {
                "start": start,
                "end": end,
                "score": max(0, min(100, score)),
                "reason": reason,
                "rank": rank,
            }
        )

    if not normalized_selected:
        raise ValueError("LLM response did not contain valid selected clips.")

    normalized_selected.sort(key=lambda item: item["rank"])
    return {"selected_clips": normalized_selected[:5]}


def _rule_based_selection(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fallback: select top 3-5 candidates by raw_score."""
    if not candidates:
        return {"selected_clips": []}

    selected_count = min(5, max(3, len(candidates))) if len(candidates) >= 3 else len(candidates)
    top_candidates = sorted(candidates, key=lambda item: item["raw_score"], reverse=True)[:selected_count]
    highest_score = max(float(candidate["raw_score"]) for candidate in top_candidates) or 1.0

    selected_clips = []
    for rank, candidate in enumerate(top_candidates, start=1):
        normalized_score = int(round((float(candidate["raw_score"]) / highest_score) * 100.0))
        selected_clips.append(
            {
                "start": float(candidate["start"]),
                "end": float(candidate["end"]),
                "score": max(1, min(100, normalized_score)),
                "reason": "High combined engagement score",
                "rank": rank,
            }
        )

    return {"selected_clips": selected_clips}


def select_clips(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select final clips from candidate clips using local Ollama with fallback.

    Expected input:
    {
      "candidates": [{"start": 0.0, "end": 30.0, "raw_score": 10.0, "signals": {...}}],
      "segments": [{"start": 0.0, "end": 2.0, "text": "..."}]
    }
    """
    try:
        candidates = _validate_candidates(input_data.get("candidates"))
        segments = _validate_segments(input_data.get("segments"))

        if not candidates:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="No valid candidate clips were provided.",
            )

        config = load_config()
        model_name = str(_config_value(config, "llm_model_name"))
        ollama_host = str(_config_value(config, "ollama_host"))

        llm_errors = []
        for attempt in range(2):
            strict_retry = attempt == 1
            try:
                user_prompt = _build_user_prompt(candidates, segments, strict_retry=strict_retry)
                raw_response = _call_ollama(
                    model_name=model_name,
                    ollama_host=ollama_host,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                parsed = _parse_llm_json(raw_response)
                logger.info("Director LLM selection completed successfully: %s", parsed)
                return build_agent_response(
                    agent=AGENT_NAME,
                    status="success",
                    data=parsed,
                    confidence=0.95,
                    error=None,
                )
            except Exception as exc:
                error_message = str(exc)
                llm_errors.append(error_message)
                logger.warning("Director LLM attempt %s failed: %s", attempt + 1, error_message)

        fallback_data = _rule_based_selection(candidates)
        if not fallback_data["selected_clips"]:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="LLM failed and rule-based fallback could not select clips.",
            )

        logger.info("Using rule-based Director fallback: %s", fallback_data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="partial",
            data=fallback_data,
            confidence=0.6,
            error="LLM selection failed; used rule-based fallback. Last error: " + llm_errors[-1],
        )

    except Exception as exc:
        logger.exception("Director Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Director Agent.")
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--transcript-json", required=True)
    args = parser.parse_args()

    with Path(args.candidates_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        candidates_response = json.load(file)
    with Path(args.transcript_json).expanduser().resolve().open("r", encoding="utf-8") as file:
        transcript_data = json.load(file)

    candidates_data = candidates_response.get("data", candidates_response)
    response = select_clips(
        {
            "candidates": candidates_data.get("candidates", []),
            "segments": transcript_data.get("segments", []),
        }
    )

    print(json.dumps(response, indent=2, ensure_ascii=False))
