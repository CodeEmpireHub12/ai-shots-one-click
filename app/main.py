"""Simple Project Raven end-to-end orchestrator.

Runs the local pipeline from input normalization through Stage 9 quality
checks, checkpointing project_state.json after every stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from raven.analyzer.clip_finder_agent import find_candidate_clips
from raven.analyzer.keyword_agent import analyze_keywords
from raven.analyzer.scene_agent import analyze_scenes
from raven.analyzer.speech_agent import analyze_speech
from raven.app.input_normalizer import normalize_input
from raven.app.preprocessor import preprocess_video
from raven.app.project_manager import (
    consolidate_project_logs,
    copy_deliverable,
    update_project_state,
)
from raven.caption.caption_agent import generate_captions
from raven.common.schema import build_agent_response
from raven.director.director_agent import select_clips
from raven.exporter.export_agent import export_video
from raven.exporter.timeline_agent import build_timeline
from raven.metadata.metadata_agent import generate_metadata
from raven.quality.quality_agent import check_quality
from raven.transcriber.transcript_agent import transcribe_audio


AGENT_NAME = "main_orchestrator"


def _save_json(path: Path, data: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return str(path)


ProgressCallback = Optional[Callable[[str, Dict[str, Any]], None]]


def _emit_progress(progress_callback: ProgressCallback, stage_name: str, response: Dict[str, Any]) -> None:
    """Send a standard envelope update to the UI, if a callback is provided."""
    if progress_callback:
        progress_callback(stage_name, response)


def _running_response(stage_name: str) -> Dict[str, Any]:
    """Build a UI-only running envelope for live progress display."""
    return build_agent_response(
        agent=stage_name,
        status="partial",
        data={"ui_status": "running"},
        confidence=0.0,
        error=None,
    )


def _checkpoint(project_id: str, stage_name: str, response: Dict[str, Any], output_path: str | None = None) -> None:
    update_project_state(
        project_id=project_id,
        last_completed_stage=stage_name,
        stage_response=response,
        output_path=output_path,
    )


def _raise_if_failed(response: Dict[str, Any], stage_name: str, progress_callback: ProgressCallback = None) -> None:
    if response.get("status") == "failed":
        _emit_progress(progress_callback, stage_name, response)
        raise RuntimeError(f"{stage_name} failed: {response.get('error')}")


def _load_project_input(project_dir: Path) -> Dict[str, Any]:
    path = project_dir / "normalized_input.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _clip_text(clip: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
    texts = []
    clip_start = float(clip["start"])
    clip_end = float(clip["end"])
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if max(clip_start, start) < min(clip_end, end):
            texts.append(str(segment.get("text", "")))
    return " ".join(texts).strip()


def run_full_pipeline(input_data: Dict[str, Any], progress_callback: ProgressCallback = None) -> Dict[str, Any]:
    """
    Run Project Raven pipeline through Stage 9.

    Expected input:
    {"youtube_url": "https://..."}
    or
    {"file_path": "/path/to/video.mp4"}
    """
    project_id = None
    project_dir = None

    try:
        # Stage 2: normalize input / download if needed.
        _emit_progress(progress_callback, "input_normalizer", _running_response("input_normalizer"))
        normalizer_response = normalize_input(input_data)
        _raise_if_failed(normalizer_response, "input_normalizer", progress_callback)
        normalized = normalizer_response["data"]
        project_id = normalized["project_id"]
        project_dir = Path(normalized["video_path"]).resolve().parent
        normalizer_output_path = _save_json(project_dir / "normalizer_output.json", normalizer_response)
        _checkpoint(project_id, "input_normalizer", normalizer_response, normalizer_output_path)
        _emit_progress(progress_callback, "input_normalizer", normalizer_response)

        # Stage 3: preprocess + transcript.
        _emit_progress(progress_callback, "preprocessor", _running_response("preprocessor"))
        preprocessor_response = preprocess_video({"video_path": normalized["video_path"]})
        _raise_if_failed(preprocessor_response, "preprocessor", progress_callback)
        preprocessor_output_path = _save_json(project_dir / "preprocessor_output.json", preprocessor_response)
        _checkpoint(project_id, "preprocessor", preprocessor_response, preprocessor_output_path)
        _emit_progress(progress_callback, "preprocessor", preprocessor_response)

        project_input = _load_project_input(project_dir)
        captions_path = (project_input.get("downloader") or {}).get("captions_path")
        _emit_progress(progress_callback, "transcript_agent", _running_response("transcript_agent"))
        transcript_response = transcribe_audio(
            {
                "audio_path": preprocessor_response["data"]["audio_path"],
                "youtube_captions_available": normalized.get("youtube_captions_available", False),
                "captions_path": captions_path,
            }
        )
        _raise_if_failed(transcript_response, "transcript_agent", progress_callback)
        transcript_output_path = _save_json(project_dir / "transcript_output.json", transcript_response)
        _checkpoint(project_id, "transcript_agent", transcript_response, transcript_output_path)
        _emit_progress(progress_callback, "transcript_agent", transcript_response)
        segments = transcript_response["data"]["segments"]

        # Stage 4: scene + speech analysis.
        _emit_progress(progress_callback, "scene_agent", _running_response("scene_agent"))
        scene_response = analyze_scenes({"video_path": normalized["video_path"]})
        _raise_if_failed(scene_response, "scene_agent", progress_callback)
        scene_output_path = _save_json(project_dir / "scene_output.json", scene_response)
        _checkpoint(project_id, "scene_agent", scene_response, scene_output_path)
        _emit_progress(progress_callback, "scene_agent", scene_response)

        _emit_progress(progress_callback, "speech_agent", _running_response("speech_agent"))
        speech_response = analyze_speech({"audio_path": preprocessor_response["data"]["audio_path"], "segments": segments})
        _raise_if_failed(speech_response, "speech_agent", progress_callback)
        speech_output_path = _save_json(project_dir / "speech_output.json", speech_response)
        _checkpoint(project_id, "speech_agent", speech_response, speech_output_path)
        _emit_progress(progress_callback, "speech_agent", speech_response)

        # Stage 5: keyword + candidate clips.
        _emit_progress(progress_callback, "keyword_agent", _running_response("keyword_agent"))
        keyword_response = analyze_keywords({"segments": segments})
        _raise_if_failed(keyword_response, "keyword_agent", progress_callback)
        keyword_output_path = _save_json(project_dir / "keyword_output.json", keyword_response)
        _checkpoint(project_id, "keyword_agent", keyword_response, keyword_output_path)
        _emit_progress(progress_callback, "keyword_agent", keyword_response)

        _emit_progress(progress_callback, "clip_finder_agent", _running_response("clip_finder_agent"))
        clip_finder_response = find_candidate_clips(
            {
                "scenes": scene_response["data"].get("scenes", []),
                "segment_features": speech_response["data"].get("segment_features", []),
                "hooks": keyword_response["data"].get("hooks", []),
                "questions": keyword_response["data"].get("questions", []),
                "keywords": keyword_response["data"].get("keywords", []),
            }
        )
        _raise_if_failed(clip_finder_response, "clip_finder_agent", progress_callback)
        clip_finder_output_path = _save_json(project_dir / "clip_candidates.json", clip_finder_response)
        _checkpoint(project_id, "clip_finder_agent", clip_finder_response, clip_finder_output_path)
        _emit_progress(progress_callback, "clip_finder_agent", clip_finder_response)

        # Stage 6: director selection.
        _emit_progress(progress_callback, "director_agent", _running_response("director_agent"))
        director_response = select_clips(
            {
                "candidates": clip_finder_response["data"].get("candidates", []),
                "segments": segments,
            }
        )
        _raise_if_failed(director_response, "director_agent", progress_callback)
        director_output_path = _save_json(project_dir / "director_selected_clips.json", director_response)
        _checkpoint(project_id, "director_agent", director_response, director_output_path)
        _emit_progress(progress_callback, "director_agent", director_response)

        selected_clips = director_response["data"].get("selected_clips", [])
        if not selected_clips:
            raise RuntimeError("Director produced no selected clips.")

        final_outputs = []
        original_title = project_input.get("original_title") or normalized.get("original_title")

        # Stages 7-9 for each selected clip.
        for index, selected_clip in enumerate(selected_clips, start=1):
            clip_with_video = dict(selected_clip)
            clip_with_video["video_path"] = normalized["video_path"]

            _emit_progress(progress_callback, "caption_agent", _running_response("caption_agent"))
            caption_response = generate_captions(
                {
                    "clip": selected_clip,
                    "segments": segments,
                    "project_dir": str(project_dir),
                    "clip_index": index,
                }
            )
            _raise_if_failed(caption_response, "caption_agent", progress_callback)
            caption_output_path = _save_json(project_dir / f"clips/clip_{index}/caption_output.json", caption_response)
            _checkpoint(project_id, "caption_agent", caption_response, caption_output_path)
            _emit_progress(progress_callback, "caption_agent", caption_response)

            text = _clip_text(selected_clip, segments)
            _emit_progress(progress_callback, "metadata_agent", _running_response("metadata_agent"))
            metadata_response = generate_metadata({"clip_transcript_text": text, "original_title": original_title})
            _raise_if_failed(metadata_response, "metadata_agent", progress_callback)
            metadata_output_path = _save_json(project_dir / f"clips/clip_{index}/metadata_output.json", metadata_response)
            _checkpoint(project_id, "metadata_agent", metadata_response, metadata_output_path)
            _emit_progress(progress_callback, "metadata_agent", metadata_response)

            _emit_progress(progress_callback, "timeline_agent", _running_response("timeline_agent"))
            timeline_response = build_timeline(
                {
                    "clip": clip_with_video,
                    "ass_path": caption_response["data"]["ass_path"],
                    "clip_index": index,
                }
            )
            _raise_if_failed(timeline_response, "timeline_agent", progress_callback)
            timeline_output_path = _save_json(project_dir / f"clips/clip_{index}/timeline_output.json", timeline_response)
            _checkpoint(project_id, "timeline_agent", timeline_response, timeline_output_path)
            _emit_progress(progress_callback, "timeline_agent", timeline_response)

            _emit_progress(progress_callback, "export_agent", _running_response("export_agent"))
            export_response = export_video(
                {
                    "ffmpeg_command": timeline_response["data"]["ffmpeg_command"],
                    "output_path": timeline_response["data"]["output_path"],
                }
            )
            _raise_if_failed(export_response, "export_agent", progress_callback)
            export_output_path = _save_json(project_dir / f"clips/clip_{index}/export_output.json", export_response)
            _checkpoint(project_id, "export_agent", export_response, export_output_path)
            _emit_progress(progress_callback, "export_agent", export_response)

            expected_duration = float(selected_clip["end"]) - float(selected_clip["start"])
            _emit_progress(progress_callback, "quality_agent", _running_response("quality_agent"))
            quality_response = check_quality(
                {
                    "rendered_clip_path": export_response["data"]["output_path"],
                    "expected_duration": expected_duration,
                }
            )
            _raise_if_failed(quality_response, "quality_agent", progress_callback)
            quality_output_path = _save_json(project_dir / f"clips/clip_{index}/quality_output.json", quality_response)
            _checkpoint(project_id, "quality_agent", quality_response, quality_output_path)
            _emit_progress(progress_callback, "quality_agent", quality_response)

            metadata = metadata_response["data"]
            safe_filename = metadata.get("filename") or f"clip_{index}"
            deliverable_response = copy_deliverable(
                project_id,
                export_response["data"]["output_path"],
                filename=f"{index:02d}_{safe_filename}.mp4",
            )

            final_outputs.append(
                {
                    "clip_index": index,
                    "video_path": export_response["data"]["output_path"],
                    "metadata": metadata,
                    "captions_path": caption_response["data"]["ass_path"],
                    "quality": quality_response["data"],
                    "deliverable_path": (deliverable_response.get("data") or {}).get("deliverable_path"),
                }
            )

        summary_response = consolidate_project_logs(project_id)
        summary_log_path = (summary_response.get("data") or {}).get("summary_log_path")

        data = {
            "project_id": project_id,
            "project_dir": str(project_dir),
            "final_outputs": final_outputs,
            "summary_log_path": summary_log_path,
            "project_state_path": str(project_dir / "project_state.json"),
        }
        return build_agent_response(AGENT_NAME, "success", data, 1.0, None)

    except Exception as exc:
        if project_id:
            consolidate_project_logs(project_id)
        return build_agent_response(AGENT_NAME, "failed", {"project_id": project_id, "project_dir": str(project_dir) if project_dir else None}, 0.0, str(exc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Project Raven full local pipeline.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--youtube-url", default=None)
    input_group.add_argument("--file-path", default=None)
    args = parser.parse_args()

    payload = {"youtube_url": args.youtube_url} if args.youtube_url else {"file_path": args.file_path}
    print(json.dumps(run_full_pipeline(payload), indent=2, ensure_ascii=False))
