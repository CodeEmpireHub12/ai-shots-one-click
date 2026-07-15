"""Transcript Agent for Project Raven.

Uses valid YouTube captions when available. Otherwise, transcribes local
audio with faster-whisper using the configured model size.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from raven.common.config import load_config
from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "transcript_agent"
logger = get_agent_logger(AGENT_NAME)


TIMESTAMP_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[\.,]\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}[\.,]\d{3})"
)
TAG_PATTERN = re.compile(r"<[^>]+>")
SETTING_DEFAULTS = {
    "whisper_model_size": "small",
    "whisper_fallback_model_size": "base",
    "whisper_low_confidence_threshold": -1.0,
    "whisper_silence_no_speech_threshold": 0.8,
}


def _setting(config: Dict[str, Any], key: str) -> Any:
    return config.get(key, SETTING_DEFAULTS[key])


def _timestamp_to_seconds(timestamp: str) -> float:
    """Convert SRT/VTT timestamp to float seconds."""
    clean = timestamp.replace(",", ".")
    hours, minutes, seconds = clean.split(":")
    return (float(hours) * 3600.0) + (float(minutes) * 60.0) + float(seconds)


def _seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp."""
    safe_seconds = max(0.0, float(seconds))
    hours = int(safe_seconds // 3600)
    minutes = int((safe_seconds % 3600) // 60)
    whole_seconds = int(safe_seconds % 60)
    milliseconds = int(round((safe_seconds - int(safe_seconds)) * 1000))

    if milliseconds >= 1000:
        whole_seconds += 1
        milliseconds -= 1000

    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _clean_caption_text(lines: List[str]) -> str:
    """Clean caption cue text."""
    text = " ".join(line.strip() for line in lines if line.strip())
    text = TAG_PATTERN.sub("", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_caption_file(captions_path: Path) -> List[Dict[str, Any]]:
    """Parse .vtt or .srt captions into Raven segment format."""
    content = captions_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    segments: List[Dict[str, Any]] = []
    index = 0

    while index < len(lines):
        line = lines[index].strip().lstrip("\ufeff")
        match = TIMESTAMP_PATTERN.search(line)

        if not match:
            index += 1
            continue

        start = _timestamp_to_seconds(match.group("start"))
        end = _timestamp_to_seconds(match.group("end"))
        index += 1

        text_lines = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index])
            index += 1

        text = _clean_caption_text(text_lines)
        if text and end > start:
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": text,
                }
            )

        index += 1

    return segments


def _save_srt(segments: List[Dict[str, Any]], srt_path: Path) -> None:
    """Save segments as transcript.srt."""
    lines = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{_seconds_to_srt_timestamp(segment['start'])} --> "
            f"{_seconds_to_srt_timestamp(segment['end'])}"
        )
        lines.append(str(segment["text"]))
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")


def _save_transcript_json(data: Dict[str, Any], transcript_json_path: Path) -> None:
    """Save transcript JSON data into the project folder."""
    with transcript_json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _get_project_dir(audio_path: Optional[str], captions_path: Optional[str]) -> Path:
    """Infer project folder from audio_path or captions_path."""
    if audio_path:
        return Path(audio_path).expanduser().resolve().parent
    if captions_path:
        return Path(captions_path).expanduser().resolve().parent
    return Path.cwd()


def _is_audio_silent(audio_path: Path, no_speech_threshold: float) -> bool:
    """
    Basic silence check for WAV audio.

    This is a lightweight guard before Whisper. It only checks WAV files.
    If it cannot read the file, it returns False and lets Whisper try.
    """
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            if frame_count == 0:
                return True

            sample_width = wav_file.getsampwidth()
            if sample_width != 2:
                return False

            frames = wav_file.readframes(frame_count)
            if not frames:
                return True

            import audioop

            rms = audioop.rms(frames, sample_width)
            normalized_rms = rms / 32768.0
            silence_cutoff = 1.0 - float(no_speech_threshold)
            return normalized_rms <= silence_cutoff
    except Exception as exc:
        logger.warning("Audio silence check skipped: %s", exc)
        return False


def _run_faster_whisper(
    audio_path: Path,
    model_size: str,
) -> Tuple[List[Dict[str, Any]], Optional[str], float]:
    """Run faster-whisper and return segments, language, and confidence score."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install it before transcribing audio without captions."
        ) from exc

    logger.info("Loading faster-whisper model: %s", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    whisper_segments, info = model.transcribe(str(audio_path))

    segments: List[Dict[str, Any]] = []
    confidence_values = []

    for segment in whisper_segments:
        text = str(segment.text or "").strip()
        if not text:
            continue

        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            }
        )

        avg_logprob = getattr(segment, "avg_logprob", None)
        if avg_logprob is not None:
            confidence_values.append(float(avg_logprob))

    language = getattr(info, "language", None)
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    return segments, language, float(confidence)


def _transcribe_with_retry(audio_path: Path, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """Run faster-whisper, retrying once with fallback model when needed."""
    primary_model_size = str(_setting(config, "whisper_model_size"))
    fallback_model_size = str(_setting(config, "whisper_fallback_model_size"))
    low_confidence_threshold = float(_setting(config, "whisper_low_confidence_threshold"))
    no_speech_threshold = float(_setting(config, "whisper_silence_no_speech_threshold"))

    should_retry = _is_audio_silent(audio_path, no_speech_threshold)
    if should_retry:
        logger.warning("Audio appears silent. Primary transcription will still be attempted before fallback.")

    segments, language, confidence = _run_faster_whisper(audio_path, primary_model_size)

    if should_retry or not segments or confidence < low_confidence_threshold:
        logger.warning(
            "Retrying transcription with fallback model=%s. segments=%s confidence=%s",
            fallback_model_size,
            len(segments),
            confidence,
        )
        segments, language, _ = _run_faster_whisper(audio_path, fallback_model_size)

    return segments, language or "unknown"


def transcribe_audio(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build transcript from captions or audio.

    Expected input:
    {
      "audio_path": "/path/to/audio.wav",
      "youtube_captions_available": true,
      "captions_path": "/path/to/source.en.vtt"
    }
    """
    try:
        audio_path_value = input_data.get("audio_path")
        captions_path_value = input_data.get("captions_path")
        youtube_captions_available = bool(input_data.get("youtube_captions_available", False))

        project_dir = _get_project_dir(audio_path_value, captions_path_value)
        project_dir.mkdir(parents=True, exist_ok=True)
        srt_path = project_dir / "transcript.srt"
        transcript_json_path = project_dir / "transcript.json"

        segments: List[Dict[str, Any]] = []
        language_detected = "unknown"
        transcript_source = "whisper"

        if youtube_captions_available and captions_path_value:
            captions_path = Path(str(captions_path_value)).expanduser().resolve()
            if captions_path.exists() and captions_path.is_file():
                parsed_segments = _parse_caption_file(captions_path)
                if parsed_segments:
                    logger.info("Using valid YouTube captions: %s", captions_path)
                    segments = parsed_segments
                    language_detected = "caption_file"
                    transcript_source = "youtube_captions"
                else:
                    logger.warning("Caption file existed but had no valid segments: %s", captions_path)
            else:
                logger.warning("Caption path is invalid: %s", captions_path)

        if not segments:
            if not audio_path_value:
                return build_agent_response(
                    agent=AGENT_NAME,
                    status="failed",
                    data={},
                    confidence=0.0,
                    error="audio_path is required when valid captions are unavailable.",
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

            config = load_config()
            segments, language_detected = _transcribe_with_retry(audio_path, config)

        if not segments:
            return build_agent_response(
                agent=AGENT_NAME,
                status="failed",
                data={},
                confidence=0.0,
                error="No transcript segments were produced.",
            )

        data = {
            "segments": segments,
            "srt_path": str(srt_path),
            "language_detected": language_detected,
            "transcript_json_path": str(transcript_json_path),
            "transcript_source": transcript_source,
        }

        _save_srt(segments, srt_path)
        _save_transcript_json(data, transcript_json_path)

        logger.info("Transcript completed successfully: %s", data)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data=data,
            confidence=1.0,
            error=None,
        )

    except Exception as exc:
        logger.exception("Transcript Agent failed.")
        return build_agent_response(
            agent=AGENT_NAME,
            status="failed",
            data={},
            confidence=0.0,
            error=str(exc),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven Transcript Agent.")
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--youtube-captions-available", action="store_true")
    parser.add_argument("--captions-path", default=None)
    args = parser.parse_args()

    print(
        json.dumps(
            transcribe_audio(
                {
                    "audio_path": args.audio_path,
                    "youtube_captions_available": args.youtube_captions_available,
                    "captions_path": args.captions_path,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
