# Project Raven Local UI

Project Raven uses a minimal Streamlit UI for local-only operation.

## Run

From the workspace root:

```bash
streamlit run raven/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

## Notes

- No login/auth.
- No cloud server.
- UI is a thin layer over `raven.app.main.run_full_pipeline` and Project Manager functions.
- UI does not call FFmpeg, Whisper, Ollama, OpenCV, or media tools directly.
- Settings panel edits `raven/settings/config.yaml`.
- Progress uses the standard Project Raven response envelope from the orchestrator.
