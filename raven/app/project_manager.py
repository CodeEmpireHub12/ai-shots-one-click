"""Project Manager Agent for Project Raven.

Maintains project_state.json, resume information, and per-project summary
logs. This module follows the standard agent response envelope.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from raven.common.logger import get_agent_logger
from raven.common.schema import build_agent_response


AGENT_NAME = "project_manager"
logger = get_agent_logger(AGENT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = PROJECT_ROOT / "projects"
GLOBAL_LOGS_DIR = PROJECT_ROOT / "logs"

STAGE_ORDER = [
    "input_normalizer",
    "preprocessor",
    "transcript_agent",
    "scene_agent",
    "speech_agent",
    "keyword_agent",
    "clip_finder_agent",
    "director_agent",
    "caption_agent",
    "metadata_agent",
    "timeline_agent",
    "export_agent",
    "quality_agent",
]


def _project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def _project_state_path(project_id: str) -> Path:
    return _project_dir(project_id) / "project_state.json"


def _safe_summary(response: Dict[str, Any]) -> Dict[str, Any]:
    """Create a small summary from a stage response."""
    data = response.get("data", {}) if isinstance(response, dict) else {}
    data_keys = sorted(data.keys()) if isinstance(data, dict) else []
    return {
        "agent": response.get("agent") if isinstance(response, dict) else None,
        "status": response.get("status") if isinstance(response, dict) else None,
        "confidence": response.get("confidence") if isinstance(response, dict) else None,
        "error": response.get("error") if isinstance(response, dict) else None,
        "data_keys": data_keys,
    }


def _load_state(project_id: str) -> Dict[str, Any]:
    state_path = _project_state_path(project_id)
    if not state_path.exists():
        return {
            "project_id": project_id,
            "last_completed_stage": None,
            "timestamp": None,
            "stage_outputs_summary": {},
        }
    with state_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_state(project_id: str, state: Dict[str, Any]) -> Path:
    project_dir = _project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    state_path = _project_state_path(project_id)
    with state_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, ensure_ascii=False)
    return state_path


def update_project_state(
    project_id: str,
    last_completed_stage: str,
    stage_response: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Write/update projects/<id>/project_state.json after a stage."""
    try:
        state = _load_state(project_id)
        state["project_id"] = project_id
        state["last_completed_stage"] = last_completed_stage
        state["timestamp"] = float(time.time())
        state.setdefault("stage_outputs_summary", {})
        state["stage_outputs_summary"][last_completed_stage] = {
            **_safe_summary(stage_response),
            "output_path": output_path,
            "timestamp": float(time.time()),
        }
        state_path = _write_state(project_id, state)
        logger.info("Project state updated: project_id=%s stage=%s", project_id, last_completed_stage)
        return build_agent_response(
            agent=AGENT_NAME,
            status="success",
            data={"project_state_path": str(state_path), "last_completed_stage": last_completed_stage},
            confidence=1.0,
            error=None,
        )
    except Exception as exc:
        logger.exception("Project state update failed.")
        return build_agent_response(AGENT_NAME, "failed", {}, 0.0, str(exc))


def resume_project(project_id: str) -> Dict[str, Any]:
    """Read project_state.json and determine the next stage to resume."""
    try:
        state_path = _project_state_path(project_id)
        if not state_path.exists():
            return build_agent_response(
                AGENT_NAME,
                "failed",
                {},
                0.0,
                f"project_state.json not found for project_id={project_id}",
            )

        state = _load_state(project_id)
        last_completed_stage = state.get("last_completed_stage")
        if last_completed_stage in STAGE_ORDER:
            next_index = STAGE_ORDER.index(last_completed_stage) + 1
            next_stage = STAGE_ORDER[next_index] if next_index < len(STAGE_ORDER) else None
        else:
            next_stage = STAGE_ORDER[0]

        data = {
            "project_id": project_id,
            "last_completed_stage": last_completed_stage,
            "next_stage": next_stage,
            "skip_completed_stages": STAGE_ORDER[: STAGE_ORDER.index(last_completed_stage) + 1]
            if last_completed_stage in STAGE_ORDER
            else [],
            "project_state_path": str(state_path),
        }
        logger.info("Resume info loaded: %s", data)
        return build_agent_response(AGENT_NAME, "success", data, 1.0, None)
    except Exception as exc:
        logger.exception("Resume project failed.")
        return build_agent_response(AGENT_NAME, "failed", {}, 0.0, str(exc))


def consolidate_project_logs(project_id: str, agent_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Consolidate individual agent logs into projects/<id>/logs/summary.log."""
    try:
        project_dir = _project_dir(project_id)
        project_logs_dir = project_dir / "logs"
        project_logs_dir.mkdir(parents=True, exist_ok=True)
        summary_log_path = project_logs_dir / "summary.log"

        names = agent_names or STAGE_ORDER
        with summary_log_path.open("w", encoding="utf-8") as summary_file:
            summary_file.write(f"Project Raven summary log for {project_id}\n")
            summary_file.write(f"Generated at: {float(time.time())}\n\n")
            for agent_name in names:
                log_path = GLOBAL_LOGS_DIR / f"{agent_name}.log"
                summary_file.write(f"\n===== {agent_name} =====\n")
                if log_path.exists():
                    summary_file.write(log_path.read_text(encoding="utf-8", errors="replace"))
                    summary_file.write("\n")
                else:
                    summary_file.write("No log file found.\n")

        logger.info("Project logs consolidated: %s", summary_log_path)
        return build_agent_response(
            AGENT_NAME,
            "success",
            {"summary_log_path": str(summary_log_path)},
            1.0,
            None,
        )
    except Exception as exc:
        logger.exception("Consolidate project logs failed.")
        return build_agent_response(AGENT_NAME, "failed", {}, 0.0, str(exc))


def copy_deliverable(project_id: str, source_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """Copy a final artifact into projects/<id>/deliverables for convenience."""
    try:
        src = Path(source_path).expanduser().resolve()
        if not src.exists():
            return build_agent_response(AGENT_NAME, "failed", {}, 0.0, f"Source deliverable not found: {src}")
        deliverables_dir = _project_dir(project_id) / "deliverables"
        deliverables_dir.mkdir(parents=True, exist_ok=True)
        target = deliverables_dir / (filename or src.name)
        shutil.copy2(src, target)
        return build_agent_response(AGENT_NAME, "success", {"deliverable_path": str(target)}, 1.0, None)
    except Exception as exc:
        logger.exception("Copy deliverable failed.")
        return build_agent_response(AGENT_NAME, "failed", {}, 0.0, str(exc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Project Raven Project Manager Agent.")
    parser.add_argument("--resume-project-id", default=None)
    parser.add_argument("--consolidate-project-id", default=None)
    args = parser.parse_args()

    if args.resume_project_id:
        print(json.dumps(resume_project(args.resume_project_id), indent=2, ensure_ascii=False))
    elif args.consolidate_project_id:
        print(json.dumps(consolidate_project_logs(args.consolidate_project_id), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(build_agent_response(AGENT_NAME, "failed", {}, 0.0, "No action provided."), indent=2))
