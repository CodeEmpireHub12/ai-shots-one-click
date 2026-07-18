"""
Project Raven - Google Colab One Click Setup
Copy and paste this entire file into a single Colab cell and run it.
It will install everything, ask for a YouTube URL, and start generating shorts.
"""

# ============================================================
# CELL 1: PASTE THIS ENTIRE CODE IN ONE COLAB CELL AND RUN IT
# ============================================================

import os, sys, json, time, shutil, subprocess
from IPython.display import clear_output

print("=" * 60)
print("🚀 PROJECT RAVEN - ONE CLICK SHORTS GENERATOR")
print("=" * 60)

# ─── Step 1: Install dependencies ───
print("\n📦 Installing dependencies...")
!pip install -q pyyaml opencv-python moviepy numpy scipy librosa soundfile faster-whisper scenedetect yt-dlp
print("✅ Dependencies installed")

# ─── Step 2: Clone repo ───
print("\n📥 Cloning repository...")
if os.path.exists("/content/ai-shots-one-click"):
    shutil.rmtree("/content/ai-shots-one-click")
!git clone https://github.com/CodeEmpireHub12/ai-shots-one-click.git /content/ai-shots-one-click
print("✅ Repository cloned")

# ─── Step 3: Set Python path ───
sys.path.insert(0, "/content/ai-shots-one-click")
os.chdir("/content/ai-shots-one-click")

# Verify imports
import raven
import raven.app
from raven.app.main import run_full_pipeline
print("✅ Python path configured")
print(f"   raven module: {raven.__file__}")

# ─── Step 4: Get YouTube URL ───
print("\n" + "=" * 60)
print("🎬 Enter YouTube Video URL")
print("=" * 60)
video_url = input("Paste YouTube URL here: ").strip()

if not video_url:
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"⚠️ No URL entered. Using default: {video_url}")

print(f"\n🚀 Processing: {video_url}")
print("⏳ This will take several minutes (download + transcribe + analyze + export)...")
print("   Grab a coffee ☕\n")

# ─── Step 5: Run pipeline ───
start_time = time.time()
try:
    result = run_full_pipeline({"youtube_url": video_url})
    elapsed = time.time() - start_time
    
    clear_output(wait=True)
    print("=" * 60)
    print("✅ PIPELINE COMPLETED!")
    print(f"⏱️  Time taken: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 60)
    
    if result.get("status") == "success":
        clips = result["data"]["final_outputs"]
        print(f"\n🎉 {len(clips)} short clips generated successfully!")
        
        for i, clip in enumerate(clips, 1):
            path = clip.get("deliverable_path") or clip.get("video_path", "")
            meta = clip.get("metadata", {})
            title = meta.get("title", f"Clip {i}")
            print(f"\n  📹 Clip #{i}: {title}")
            print(f"     📁 Path: {path}")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"     📦 Size: {size_mb:.1f} MB")
        
        # Download all clips
        print("\n📥 Downloading clips to your computer...")
        from google.colab import files
        for clip in clips:
            path = clip.get("deliverable_path") or clip.get("video_path", "")
            if path and os.path.exists(path):
                files.download(path)
                time.sleep(1)
        
        print("\n✅ All clips downloaded!")
    else:
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🏁 Done! Run this cell again for another video.")
print("=" * 60)