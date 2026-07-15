"""Speech Agent for Project Raven.

Computes deterministic speech features from transcript segments and audio:
pace, pause_before, and RMS energy per transcript segment.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "speech_agent"
logger = get_agent_logger(AGENT_NAME)


DEFAULTS = {
    "speech_audio_sample_rate": 16000,
}
WORD_PATTERN = re.compile(r"\b[\w']+\b")


def _config_value(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, DEFAULTS[key])


def _count_words(text: str) -> int:
    """Count words in transcript text."""
    return len(WORD_PATTERN.findall(text or ""))


def _segment_energy(audio: np.ndarray, sample_rate: int, start: float, end: float) -> float:
    """Compute normalized RMS energy for a segment."""
    safe_start = max(0.0, float(start))
    safe_end = max(safe_start, float(end))
    start_sample = int(round(safe_start * sample_rate))
    end_sample = int(round(safe_end * sample_rate))
    start_sample = min(start_sample, len(audio))
    end_sample = min(end_sample, len(audio))

    if end_sample <= start_sample:
        return 0.0

    segment_audio = audio[start_sample:end_sample]
    if segment_audio.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(segment_audio))))
    return rms


def _validate_segments(segments: Any) -> List[Dict[str, Any]]:
    """Validate and normalize transcript segment input."""
    if not isinstance(segments, list):
        raise ValueError("segments must be a list.")

    normalized_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue

        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text = str(segment.get("text", ""))

        if end <= start:
            continue

        normalized_segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    normalized_segments.sort(key=lambda item: item["start"])
    return normalized_segments


def analyze_speech(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze transcript/audio speech features.

    Expected input:
    {
      "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
      "audio_path": "/path/to/audio.wav"
    }
    """
    try:
        audio_path_value = input_data.get("audio_path")
        if not audio_path_value:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="audio_path is required.",
            )

        audio_path = Path(str(audio_path_value)).expanduser().resolve()
        if not audio_path.exists() or not audio_path.is_file():
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error=f"Audio file does not exist: {audio_path}",
            )

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
        sample_rate = int(_config_value(config, "speech_audio_sample_rate"))
        audio, loaded_sample_rate = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        sample_rate = int(loaded_sample_rate)

        segment_features = []
        previous_end = None

        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            duration = max(0.001, end - start)
            word_count = _count_words(segment.get("text", ""))
            pace = float(word_count / duration)
            pause_before = float(max(0.0, start - previous_end)) if previous_end is not None else 0.0
            energy = _segment_energy(audio, sample_rate, start, end)

            segment_features.append(
                {
                    "start": start,
                    "end": end,
                    "energy": float(energy),
                    "pace": pace,
                    "pause_before": pause_before,
                }
            )

            previous_end = end

        data = {"segment_features": segment_features}
        logger.info("Speech analysis completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Speech Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Speech Agent.")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--transcript-json", required=True)
    args = parser.parse_args()

    transcript_path = Path(args.transcript_json).expanduser().resolve()
    with transcript_path.open("r", encoding="utf-8") as file:
        transcript_data = json.load(file)

    print(
        json.dumps(
            analyze_speech(
                {
                    "audio_path": args.audio_path,
                    "segments": transcript_data.get("segments", []),
                }
            ),
            indent=2,
        )
    )
