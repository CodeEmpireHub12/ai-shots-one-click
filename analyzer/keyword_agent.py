"""Keyword Agent for Project Raven.

Extracts hooks, questions, and lightweight TF-IDF keywords from transcript
segments. This agent does not use heavy models or cloud APIs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "keyword_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "keyword_top_n": 10,
}

INTERROGATIVE_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "whose",
    "whom",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
}

HOOK_PATTERNS = [
    re.compile(r"^\s*you won['’]?t believe\b", re.IGNORECASE),
    re.compile(r"^\s*here['’]?s why\b", re.IGNORECASE),
    re.compile(r"^\s*here is why\b", re.IGNORECASE),
    re.compile(r"^\s*this is why\b", re.IGNORECASE),
    re.compile(r"^\s*the truth about\b", re.IGNORECASE),
    re.compile(r"^\s*what nobody tells you\b", re.IGNORECASE),
    re.compile(r"^\s*\d+\s+(ways|tips|tricks|reasons|steps|things)\s+to\b", re.IGNORECASE),
]

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "us",
    "was", "we", "were", "with", "you", "your", "all", "about", "into",
    "out", "up", "down", "really", "pretty", "much", "just", "very", "here",
    "thing", "things", "guys", "guy", "say",
}

WORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z']{1,}\b")
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?", re.MULTILINE)


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _validate_segments(segments: Any) -> List[Dict[str, Any]]:
    """Validate transcript segments and normalize timestamps to floats."""
    if not isinstance(segments, list):
        raise ValueError("segments must be a list.")

    normalized_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue

        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text = str(segment.get("text", "")).strip()

        if end <= start or not text:
            continue

        normalized_segments.append({"start": start, "end": end, "text": text})

    normalized_segments.sort(key=lambda item: item["start"])
    return normalized_segments


def _tokenize(text: str) -> List[str]:
    """Tokenize text for lightweight keyword extraction."""
    tokens = [match.group(0).lower().strip("'") for match in WORD_PATTERN.finditer(text)]
    return [token for token in tokens if token and token not in STOP_WORDS]


def _is_question(text: str) -> bool:
    """Detect a question by punctuation or interrogative first word."""
    clean_text = text.strip()
    if clean_text.endswith("?"):
        return True

    tokens = WORD_PATTERN.findall(clean_text.lower())
    return bool(tokens and tokens[0] in INTERROGATIVE_WORDS)


def _is_hook(text: str) -> bool:
    """Detect simple hook phrases."""
    clean_text = text.strip()
    if any(pattern.search(clean_text) for pattern in HOOK_PATTERNS):
        return True

    tokens = WORD_PATTERN.findall(clean_text.lower())
    if not tokens:
        return False

    # Lightweight fallback: very short segments beginning with attention words.
    attention_starts = {"look", "listen", "watch", "imagine", "remember"}
    return tokens[0] in attention_starts and len(tokens) <= 12


def _extract_questions_and_hooks(segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract hook and question spans from transcript segments."""
    hooks = []
    questions = []

    for segment in segments:
        text = segment["text"]
        sentences = [part.group(0).strip() for part in SENTENCE_PATTERN.finditer(text) if part.group(0).strip()]
        if not sentences:
            sentences = [text]

        for sentence in sentences:
            item = {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": sentence,
            }
            if _is_question(sentence):
                questions.append(item)
            if _is_hook(sentence):
                hooks.append(item)

    return {"hooks": hooks, "questions": questions}


def _extract_tfidf_keywords(segments: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """Extract top keywords with simple TF-IDF over transcript segments."""
    documents = [_tokenize(segment["text"]) for segment in segments]
    documents = [tokens for tokens in documents if tokens]
    if not documents:
        return []

    doc_count = len(documents)
    document_frequency = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))

    scores = defaultdict(float)
    occurrence_ranges = defaultdict(lambda: {"start": None, "end": None})

    for segment, tokens in zip(segments, [_tokenize(segment["text"]) for segment in segments]):
        if not tokens:
            continue

        term_frequency = Counter(tokens)
        token_total = sum(term_frequency.values())
        for token, count in term_frequency.items():
            tf = count / token_total
            idf = math.log((doc_count + 1) / (document_frequency[token] + 1)) + 1.0
            scores[token] += tf * idf

            range_data = occurrence_ranges[token]
            if range_data["start"] is None or segment["start"] < range_data["start"]:
                range_data["start"] = float(segment["start"])
            if range_data["end"] is None or segment["end"] > range_data["end"]:
                range_data["end"] = float(segment["end"])

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    keywords = []
    for text, score in ranked:
        range_data = occurrence_ranges[text]
        keywords.append(
            {
                "text": text,
                "score": float(score),
                "start": float(range_data["start"] or 0.0),
                "end": float(range_data["end"] or 0.0),
            }
        )

    return keywords


def analyze_keywords(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract hooks, questions, and keywords from transcript segments.

    Expected input:
    {"segments": [{"start": 0.0, "end": 1.0, "text": "..."}]}
    """
    try:
        segments = _validate_segments(input_data.get("segments"))
        if not segments:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="No valid transcript segments were provided.",
            )

        config = load_config()
        top_n = int(_config_value(config, "keyword_top_n"))

        extracted = _extract_questions_and_hooks(segments)
        keywords = _extract_tfidf_keywords(segments, top_n)

        data = {
            "hooks": extracted["hooks"],
            "questions": extracted["questions"],
            "keywords": keywords,
        }

        logger.info("Keyword analysis completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Keyword Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Keyword Agent.")
    parser.add_argument("--transcript-json", required=True)
    args = parser.parse_args()

    transcript_path = Path(args.transcript_json).expanduser().resolve()
    with transcript_path.open("r", encoding="utf-8") as file:
        transcript_data = json.load(file)

    print(
        json.dumps(
            analyze_keywords({"segments": transcript_data.get("segments", [])}),
            indent=2,
            ensure_ascii=False,
        )
    )
