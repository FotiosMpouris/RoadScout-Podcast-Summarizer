# RoadScout — Podcast/YouTube Summarizer with One-Click Audio

RoadScout turns long podcasts and YouTube videos into a clean, persona-driven markdown summary plus a single stitched MP3 you can stream or download for the commute.

-   **Auto-transcript (YouTube):** Tries `youtube-transcript-api` first, then falls back to `yt-dlp` + VTT.
-   **Smart summarization:** Per-chunk + merge/fitting to a target read time (defaults to `gpt-4.1`).
-   **One MP3:** Fast, parallel TTS then stitched in-memory to a single file (no “Part 1…N” clutter).
-   **Push Notifications:** Optional, real-time alerts sent to your phone via the free `ntfy.sh` service when a summary is complete.
-   **Groovy 1970s UI:** Bright gradient canvas + dark sidebar.
-   **Quality of life:** Recognizable filenames and a detailed diagnostics panel.

## Quickstart (Streamlit Community Cloud)

#### 1. Deploy the app

Click “New app” → point to this repo → `app.py`.

#### 2. Add secrets

In Streamlit Cloud: App → Settings → Secrets.
Paste your OpenAI API key:

```toml
# .streamlit/secrets.toml (Cloud UI only)
OPENAI_API_KEY = "sk-...your key..."
You can also set OPENAI_API_KEY as an environment variable locally.
Dependencies
Make sure these two files exist at the repo root:
requirements.txt
code
Code
streamlit>=1.28
openai>=1.0
youtube-transcript-api>=0.6.2
yt-dlp>=2024.10.7
requests>=2.31
pydub>=0.25.1
packages.txt
(Installs ffmpeg in Streamlit Cloud for MP3 stitching)
code
Code
ffmpeg
Run
Open the deployed app, paste a YouTube URL (or a transcript), hit “Summarize & Play ▶️”.
Running Locally
code
Bash
# 1) Clone repo, then create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set your API key
export OPENAI_API_KEY="sk-..."   # On Windows PowerShell:  $env:OPENAI_API_KEY="sk-..."

# 4) Ensure ffmpeg is installed and available in your system's PATH
# macOS (Homebrew):   brew install ffmpeg
# Ubuntu/Debian:      sudo apt-get update && sudo apt-get install -y ffmpeg
# Windows:            Install from ffmpeg.org and add the bin directory to your PATH

# 5) Run the app
streamlit run app.py
Project Structure
code
Code
.
├─ app.py                 # Streamlit UI + workflow
├─ prompts.py             # build_persona_prompt(style, target_minutes, …)
├─ audio_utils.py         # tts_to_single_mp3(): parallel TTS + stitch
├─ requirements.txt
└─ packages.txt           # ffmpeg (for Streamlit Cloud)
How It Works (Pipeline)
Transcript Acquisition
Try youtube-transcript-api (official or auto-generated tracks).
If missing/blocked → yt-dlp finds a VTT caption URL → parsed to text.
Summarization
Split transcript into large chunks (reduces API calls).
Per-chunk summarize: gpt-4o-mini (if speed mode is on), else gpt-4.1.
Merge & fit: Consolidate with gpt-4.1, then fit to a target word count based on the Target read length slider.
Audio
tts_to_single_mp3 calls OpenAI's TTS model in parallel → stitches the audio parts into one MP3.
The download filename is made from the video title (slug) + timestamp.
Notification
If enabled, sends a push notification to a user-defined ntfy.sh topic upon completion.
UI Guide
Tone: “analytical and direct”, “clear and professional”, etc.
Target read length: 5–25 minutes → controls summary length and thus audio length.
Include timestamps when possible: Requests lightweight time anchors in the summary.
Optional focus areas: Comma-separated hints (e.g., regulatory risk, compute constraints).
Merge/Fit Model: The model used for the final, high-fidelity merge and length-fitting (gpt-4.1).
Speed mode: If ON, per-chunk summarization uses the faster/cheaper gpt-4o-mini.
TTS Voice: Choose an OpenAI voice (e.g., alloy, amber, sage, verse).
Send push notification when complete?: If ON, a text input appears. Enter your secret ntfy.sh topic name here to receive an instant alert on your phone when the job is finished. Requires the free ntfy mobile app.
Tips & Troubleshooting
No transcript retrieved?
Some videos disable captions or are region-blocked. The app already falls back to yt-dlp; if both fail, paste the transcript manually.
"Could not obtain a transcript" message?
Paste a transcript in the manual box and click again.
Try a different YouTube URL or remove any &t=... time fragments from the URL.
Push Notification Not Received?
Ensure the topic name in the sidebar exactly matches the one you subscribed to in the ntfy mobile app. It is case-sensitive.
Check that you are subscribed to the topic on your phone.
Verify your phone's internet connection and that notifications are enabled for the ntfy app.
Audio not produced / short audio?
Ensure packages.txt contains ffmpeg (for Cloud). Locally, verify ffmpeg is installed and on your system's PATH.
Configuration Notes
Secrets: The only secret required is OPENAI_API_KEY. In Streamlit Cloud, use the Secrets UI. Locally, set it as an environment variable.
Models:
Merge/Fit default: gpt-4.1
Per-chunk (Speed mode): gpt-4o-mini
TTS: OpenAI tts-1-hd or tts-1 via the audio.speech endpoint.
License
MIT — feel free to adapt to your needs.
Credits
Captions: youtube-transcript-api, yt-dlp
Summarization: OpenAI (gpt-4.1, gpt-4o-mini)
Text-to-speech: OpenAI
Audio stitching: pydub + ffmpeg
Notifications: ntfy.sh
UI: Streamlit (with a splash of 1970s flare)
