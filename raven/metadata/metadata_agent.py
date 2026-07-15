"""Metadata Agent for Project Raven.

Generates title, description, hashtags, and clean filename for one clip
using local Ollama. Falls back to a deterministic template when Ollama is
unavailable or returns invalid JSON.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "metadata_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "llm_model_name": "qwen3:4b",
    "ollama_host": "http://localhost:11434",
    "metadata_title_max_chars": 60,
    "metadata_hashtag_min_count": 5,
    "metadata_hashtag_max_count": 8,
}

SYSTEM_PROMPT = """You are a short-form video metadata writer.
Return ONLY valid JSON. No markdown. No code fences. No extra text.
The JSON must match exactly this structure:
{"title": "", "description": "", "hashtags": [], "filename": ""}
Rules:
- title must be catchy and max 60 characters.
- description must be 1-2 short lines.
- hashtags must contain 5-8 relevant hashtags.
- filename must be lowercase, clean, and contain no special characters except underscores.
"""

STRICT_RETRY_SUFFIX = """Return ONLY the JSON object, wrapped in nothing else.
No markdown. No code fences. No comments. No extra text before or after JSON.
Required shape: {"title": "", "description": "", "hashtags": [], "filename": ""}
"""

WORD_PATTERN = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9']*\b")
NON_FILENAME_PATTERN = re.compile(r"[^a-z0-9_]+")


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _clean_spaces(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _clean_filename(text: str) -> str:
    """Create a lowercase underscore filename stem."""
    words = WORD_PATTERN.findall(str(text).lower())
    cleaned = "_".join(words[:10])
    cleaned = NON_FILENAME_PATTERN.sub("_", cleaned).strip("_")
    return cleaned or "raven_short"


def _simple_title_from_transcript(transcript_text: str, max_chars: int) -> str:
    """Build a fallback title from the first few transcript words."""
    words = WORD_PATTERN.findall(transcript_text)
    if not words:
        return "Untitled Raven Short"

    title = " ".join(words[:8])
    title = title[:max_chars].strip()
    return title or "Untitled Raven Short"


def _fallback_hashtags(transcript_text: str, original_title: str | None, min_count: int, max_count: int) -> List[str]:
    """Generate simple fallback hashtags from transcript/title words."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "you",
        "your", "all", "here", "there", "really", "pretty", "much",
    }
    source = f"{original_title or ''} {transcript_text}"
    words = []
    seen = set()
    for word in WORD_PATTERN.findall(source.lower()):
        if word in stop_words or len(word) < 3 or word in seen:
            continue
        seen.add(word)
        words.append(word)

    hashtags = ["#" + word[:30] for word in words[:max_count]]
    defaults = ["#shorts", "#viral", "#video", "#clip", "#raven"]
    for tag in defaults:
        if len(hashtags) >= min_count:
            break
        if tag not in hashtags:
            hashtags.append(tag)

    return hashtags[:max_count]


def _fallback_metadata(transcript_text: str, original_title: str | None, config: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback metadata."""
    max_chars = int(_config_value(config, "metadata_title_max_chars"))
    min_tags = int(_config_value(config, "metadata_hashtag_min_count"))
    max_tags = int(_config_value(config, "metadata_hashtag_max_count"))

    title = _simple_title_from_transcript(transcript_text, max_chars)
    description_source = original_title or title
    description = f"A quick highlight from {description_source}."
    hashtags = _fallback_hashtags(transcript_text, original_title, min_tags, max_tags)
    filename = _clean_filename(title)

    return {
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "filename": filename,
    }


def _build_user_prompt(transcript_text: str, original_title: str | None, strict_retry: bool = False) -> str:
    """Build the local LLM user prompt."""
    prompt = {
        "task": "Generate publish-ready metadata for a vertical short clip.",
        "original_video_title": original_title,
        "clip_transcript_text": transcript_text,
        "required_output_format": {
            "title": "",
            "description": "",
            "hashtags": [],
            "filename": "",
        },
    }
    text = json.dumps(prompt, indent=2, ensure_ascii=False)
    if strict_retry:
        text = STRICT_RETRY_SUFFIX + "\n" + text
    return text


def _call_ollama(model_name: str, ollama_host: str, user_prompt: str) -> str:
    """Call local Ollama /api/generate and return raw text."""
    host = str(ollama_host).rstrip("/")
    url = f"{host}/api/generate"
    payload = {
        "model": model_name,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.4},
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


def _validate_metadata(raw_text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate LLM metadata JSON."""
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Metadata JSON must be an object.")

    max_chars = int(_config_value(config, "metadata_title_max_chars"))
    min_tags = int(_config_value(config, "metadata_hashtag_min_count"))
    max_tags = int(_config_value(config, "metadata_hashtag_max_count"))

    title = _clean_spaces(parsed.get("title", ""))[:max_chars].strip()
    description = _clean_spaces(parsed.get("description", ""))
    hashtags_raw = parsed.get("hashtags", [])
    filename = _clean_filename(parsed.get("filename", title))

    if not title:
        raise ValueError("Metadata title is empty.")
    if not description:
        raise ValueError("Metadata description is empty.")
    if not isinstance(hashtags_raw, list):
        raise ValueError("Metadata hashtags must be a list.")

    hashtags = []
    seen = set()
    for tag in hashtags_raw:
        clean_tag = str(tag).strip()
        if not clean_tag:
            continue
        if not clean_tag.startswith("#"):
            clean_tag = "#" + clean_tag
        clean_tag = "#" + re.sub(r"[^a-zA-Z0-9_]", "", clean_tag[1:])
        if len(clean_tag) <= 1 or clean_tag.lower() in seen:
            continue
        seen.add(clean_tag.lower())
        hashtags.append(clean_tag)

    if len(hashtags) < min_tags:
        raise ValueError("Metadata response did not include enough hashtags.")

    return {
        "title": title,
        "description": description,
        "hashtags": hashtags[:max_tags],
        "filename": filename,
    }


def generate_metadata(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate metadata for one clip.

    Expected input:
    {
      "clip_transcript_text": "...",
      "original_title": "Original YouTube title or null"
    }
    """
    try:
        transcript_text = _clean_spaces(input_data.get("clip_transcript_text", ""))
        original_title_raw = input_data.get("original_title")
        original_title = _clean_spaces(original_title_raw) if original_title_raw else None

        if not transcript_text:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="clip_transcript_text is required.",
            )

        config = load_config()
        model_name = str(_config_value(config, "llm_model_name"))
        ollama_host = str(_config_value(config, "ollama_host"))

        llm_errors = []
        for attempt in range(2):
            strict_retry = attempt == 1
            try:
                user_prompt = _build_user_prompt(transcript_text, original_title, strict_retry)
                raw_response = _call_ollama(model_name, ollama_host, user_prompt)
                data = _validate_metadata(raw_response, config)
                logger.info("Metadata LLM generation completed successfully: %s", data)
                return build_agent_response(
                    agent=AGENT_NAME,
                    status="success",
                    data=data,
                    confidence=0.95,
                    error=None,
                )
            except Exception as exc:
                error_message = str(exc)
                llm_errors.append(error_message)
                logger.warning("Metadata LLM attempt %s failed: %s", attempt + 1, error_message)

        fallback_data = _fallback_metadata(transcript_text, original_title, config)
        logger.info("Using metadata fallback: %s", fallback_data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="partial",
            data=fallback_data,
            confidence=0.6,
            error="LLM metadata generation failed; used fallback. Last error: " + llm_errors[-1],
        )

    except Exception as exc:
        logger.exception("Metadata Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Metadata Agent.")
    parser.add_argument("--clip-text", required=True)
    parser.add_argument("--original-title", default=None)
    args = parser.parse_args()

    print(
        json.dumps(
            generate_metadata(
                {
                    "clip_transcript_text": args.clip_text,
                    "original_title": args.original_title,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
