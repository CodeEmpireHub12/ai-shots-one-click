"""Project Raven local Streamlit UI.

Run locally:
    streamlit run raven/ui/streamlit_app.py --server.address 127.0.0.1

This UI is intentionally thin. It only calls the main orchestrator and
Project Manager functions. It does not call FFmpeg, Whisper, Ollama, or
any media/AI tool directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

# Ensure imports work when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from raven.app.main import run_full_pipeline
from raven.app.project_manager import resume_project, consolidate_project_logs


CONFIG_PATH = PROJECT_ROOT / "settings" / "config.yaml"
LOGS_DIR = PROJECT_ROOT / "logs"
UI_UPLOAD_DIR = PROJECT_ROOT / "projects" / "ui_uploads"

EDITABLE_CONFIG_KEYS = [
    "whisper_model_size",
    "llm_model_name",
    "ollama_host",
    "ffmpeg_path",
    "output_resolution",
    "max_clips",
]

STAGE_LOG_MAP = {
    "input_normalizer": "input_normalizer.log",
    "preprocessor": "preprocessor.log",
    "transcript_agent": "transcript_agent.log",
    "scene_agent": "scene_agent.log",
    "speech_agent": "speech_agent.log",
    "keyword_agent": "keyword_agent.log",
    "clip_finder_agent": "clip_finder_agent.log",
    "director_agent": "director_agent.log",
    "caption_agent": "caption_agent.log",
    "metadata_agent": "metadata_agent.log",
    "timeline_agent": "timeline_agent.log",
    "export_agent": "export_agent.log",
    "quality_agent": "quality_agent.log",
}


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_config(config: Dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def tail_log(stage_name: str, lines: int = 80) -> str:
    log_name = STAGE_LOG_MAP.get(stage_name)
    if not log_name:
        return "No log mapped for this stage yet."
    log_path = LOGS_DIR / log_name
    if not log_path.exists():
        return f"Waiting for log file: {log_path}"
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def file_size_mb(path: str) -> float:
    file_path = Path(path)
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)


def open_folder(path: str) -> None:
    """Open an output folder locally when supported by the OS."""
    folder = Path(path).expanduser().resolve()
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception:
        st.warning(f"Could not open folder automatically. Path: {folder}")


def save_uploaded_file(uploaded_file) -> str:
    UI_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    target_path = UI_UPLOAD_DIR / safe_name
    with target_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    return str(target_path)


def render_settings_panel() -> None:
    st.subheader("Settings")
    st.caption("Edits are saved to settings/config.yaml. Restart a running pipeline for changes to take effect.")
    config = load_config()

    with st.form("settings_form"):
        updated = dict(config)
        for key in EDITABLE_CONFIG_KEYS:
            current_value = config.get(key, "")
            if key == "max_clips":
                updated[key] = st.number_input(key, min_value=1, max_value=20, value=int(current_value or 5))
            else:
                updated[key] = st.text_input(key, value=str(current_value))
        if st.form_submit_button("Save settings"):
            save_config(updated)
            st.success("Settings saved.")


def render_progress_table(events: List[Dict[str, Any]]) -> None:
    if not events:
        st.info("No progress events yet.")
        return

    rows = []
    for event in events:
        response = event.get("response", {})
        rows.append(
            {
                "stage": event.get("stage"),
                "agent": response.get("agent"),
                "status": response.get("status"),
                "error": response.get("error"),
            }
        )
    st.dataframe(rows, use_container_width=True)


def run_pipeline_with_ui(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run orchestrator while updating Streamlit progress placeholders."""
    st.session_state["progress_events"] = []
    status_box = st.empty()
    progress_table_box = st.empty()
    log_box = st.empty()

    def progress_callback(stage_name: str, response: Dict[str, Any]) -> None:
        st.session_state["progress_events"].append({"stage": stage_name, "response": response})
        ui_status = "running" if (response.get("data") or {}).get("ui_status") == "running" else response.get("status")
        status_box.info(f"Current stage: {stage_name} — {ui_status}")
        with progress_table_box.container():
            render_progress_table(st.session_state["progress_events"])
        log_box.code(tail_log(stage_name), language="text")

    try:
        result = run_full_pipeline(input_payload, progress_callback=progress_callback)
        st.session_state["last_result"] = result
        if result.get("status") == "success":
            status_box.success("Pipeline finished successfully.")
        else:
            status_box.error("Pipeline failed. Check the error below and logs/.")
        return result
    except Exception:
        # UI must never show raw tracebacks to the user.
        result = {
            "agent": "streamlit_ui",
            "status": "failed",
            "data": {},
            "confidence": 0.0,
            "error": "Something went wrong, check logs/",
        }
        st.session_state["last_result"] = result
        status_box.error(result["error"])
        return result


def render_outputs(result: Dict[str, Any]) -> None:
    st.subheader("Final Outputs")
    if not result or result.get("status") != "success":
        st.info("No completed outputs yet.")
        return

    data = result.get("data", {})
    outputs = data.get("final_outputs", [])
    if not outputs:
        st.info("Pipeline succeeded but no final clips were listed.")
        return

    project_dir = data.get("project_dir")
    if project_dir and st.button("Open output folder"):
        open_folder(str(Path(project_dir) / "deliverables"))

    for output in outputs:
        video_path = output.get("deliverable_path") or output.get("video_path")
        metadata = output.get("metadata", {})
        quality = output.get("quality", {})
        st.markdown(f"### Clip {output.get('clip_index')}")
        st.write(f"**Filename:** `{Path(video_path).name if video_path else 'missing'}`")
        st.write(f"**Size:** {file_size_mb(video_path):.2f} MB" if video_path else "**Size:** unknown")
        st.write(f"**Quality:** {quality.get('overall', 'unknown')}")
        st.write(f"**Title:** {metadata.get('title', '')}")
        st.write(f"**Description:** {metadata.get('description', '')}")
        st.write("**Hashtags:** " + " ".join(metadata.get("hashtags", [])))
        if video_path and Path(video_path).exists():
            st.video(video_path)
        else:
            st.warning("Video file not found.")


def render_retry_panel() -> None:
    st.subheader("Retry / Resume")
    st.caption("Uses Project Manager resume information. The UI does not call media/AI tools directly.")
    project_id = st.text_input("Project ID to resume", value="")
    if st.button("Retry this stage"):
        if not project_id.strip():
            st.warning("Enter a project ID first.")
            return
        response = resume_project(project_id.strip())
        st.json(response)
        if response.get("status") == "success":
            next_stage = (response.get("data") or {}).get("next_stage")
            if next_stage is None:
                st.success("Project already completed all tracked stages.")
            else:
                st.info(f"Project Manager says to resume from: {next_stage}. Already-completed stages should be skipped.")
                st.warning("Minimal v1 UI identifies the retry stage from project_state.json. If the same failure repeats, check logs/summary.log before continuing.")

    if st.button("Consolidate logs for this project"):
        if project_id.strip():
            st.json(consolidate_project_logs(project_id.strip()))
        else:
            st.warning("Enter a project ID first.")


def main() -> None:
    st.set_page_config(page_title="Project Raven", layout="wide")
    st.title("Project Raven")
    st.caption("100% local/offline vertical Shorts generator. No login. No cloud.")

    with st.sidebar:
        render_settings_panel()
        st.divider()
        render_retry_panel()

    tab_start, tab_progress, tab_outputs = st.tabs(["Start", "Progress / Logs", "Outputs"])

    with tab_start:
        st.subheader("Input")
        youtube_url = st.text_input("YouTube URL", value="")
        uploaded_file = st.file_uploader("Or upload a local video", type=["mp4", "mov", "mkv"])

        if st.button("Start", type="primary"):
            if youtube_url.strip() and uploaded_file is not None:
                st.error("Provide either a YouTube URL or an uploaded file, not both.")
            elif youtube_url.strip():
                result = run_pipeline_with_ui({"youtube_url": youtube_url.strip()})
                st.json(result)
            elif uploaded_file is not None:
                saved_path = save_uploaded_file(uploaded_file)
                result = run_pipeline_with_ui({"file_path": saved_path})
                st.json(result)
            else:
                st.warning("Paste a YouTube URL or upload a local video first.")

    with tab_progress:
        st.subheader("Progress")
        render_progress_table(st.session_state.get("progress_events", []))
        if st.session_state.get("progress_events"):
            last_stage = st.session_state["progress_events"][-1]["stage"]
            st.subheader(f"Latest log: {last_stage}")
            st.code(tail_log(last_stage), language="text")

    with tab_outputs:
        render_outputs(st.session_state.get("last_result", {}))


if __name__ == "__main__":
    main()
