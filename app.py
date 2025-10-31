import os
import re
import io
from typing import List, Optional, Tuple

import streamlit as st

# ---- OpenAI client (official SDK) ----
try:
    from openai import OpenAI
except ImportError:
    st.error("Missing OpenAI SDK. Make sure 'openai' is in requirements.txt.")
    st.stop()

# ---- YouTube transcript fetcher (no ffmpeg needed) ----
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        CouldNotRetrieveTranscript,
    )
except Exception:
    YouTubeTranscriptApi = None  # degrade gracefully

# ---- Prompts (from prompts.py) ----
try:
    from prompts import build_persona_prompt
except Exception as e:
    st.error(f"Could not import prompts.py: {e}")
    st.stop()

# =========================
# App Config
# =========================
st.set_page_config(page_title="RoadScout: Podcast Summarizer", page_icon="🎧", layout="wide")
st.title("🎧 RoadScout: Podcast Summarizer")
st.caption("Paste a podcast/YouTube URL (or transcript), click once, and get a persona-driven text + audio summary.")

# =========================
# Secrets / API Key
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("No OPENAI_API_KEY detected. Add it in Streamlit → Settings → Secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Helpers
# =========================
YOUTUBE_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)

def extract_youtube_id(url: str) -> Optional[str]:
    m = YOUTUBE_REGEX.search(url.strip())
    return m.group(1) if m else None

def fetch_youtube_transcript(video_id: str, languages: Optional[List[str]] = None) -> Tuple[str, List[Tuple[float, str]]]:
    """
    Robust transcript fetcher:
    - Try official captions first (preferred)
    - Then try auto-generated captions
    - Return (joined_text, timeline [(start_seconds, text), ...])
    """
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api is not installed on this deployment.")
    languages = languages or ["en", "en-US", "en-GB"]

    try:
        # 1) Try official captions directly
        data = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except Exception:
        # 2) Fallback: search list & try generated transcripts explicitly
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        chosen = None

        # Prefer manually provided English
        for code in (languages + ["en"]):
            try:
                chosen = transcripts.find_transcript([code])
                break
            except Exception:
                continue

        # If still nothing, try auto-generated English
        if chosen is None:
            try:
                chosen = transcripts.find_generated_transcript(["en"])
            except Exception:
                pass

        if chosen is None:
            raise NoTranscriptFound("No official or auto-generated transcript available for this video.")

        data = chosen.fetch()

    timeline = [(item["start"], item["text"]) for item in data]
    joined = " ".join(seg for _, seg in timeline).strip()
    if not joined:
        # Common when YT responds with empty/HTML content
        raise CouldNotRetrieveTranscript(
            "Transcript fetch returned empty content (YouTube may be blocking or captions are disabled)."
        )
    return joined, timeline

def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    """Conservative chunking by characters; respects paragraph boundaries when possible."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    current = []
    current_len = 0
    for para in re.split(r"\n{2,}", text):
        if current_len + len(para) + 2 <= max_chars:
            current.append(para)
            current_len += len(para) + 2
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks

def summarize_transcript(transcript: str,
                         persona_prompt: str,
                         model: str = "gpt-4o-mini",
                         temperature: float = 0.2,
                         log=None) -> str:
    """
    Sends chunked transcript to the model with a system+user flow:
    - system: persona_prompt
    - user: chunk i
    Then merges with a final pass.
    """
    chunks = chunk_text(transcript, max_chars=12000)
    if log: log(f"Transcript length: {len(transcript):,} chars → {len(chunks)} chunk(s).")
    intermediates: List[str] = []

    for i, ch in enumerate(chunks, 1):
        if log: log(f"Summarizing chunk {i}/{len(chunks)} ({len(ch):,} chars)…")
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": f"TRANSCRIPT CHUNK {i}/{len(chunks)}:\n\n{ch}\n\nProvide a compact intermediate summary (bulleted where helpful)."}
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        intermediates.append(resp.choices[0].message.content.strip())

    if log: log("Merging intermediate summaries…")
    merge_messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": "Merge the following intermediate summaries into a single cohesive final report. Remove redundancies, keep structure, and ensure chronology and lenses are clear:\n\n" + "\n\n---\n\n".join(intermediates)}
    ]
    final_resp = client.chat.completions.create(
        model=model,
        messages=merge_messages,
        temperature=temperature,
    )
    return final_resp.choices[0].message.content.strip()

def tts_from_text(text: str,
                  model: str = "gpt-4o-mini-tts",
                  voice: str = "alloy",
                  max_chars: int = 8000,
                  log=None) -> bytes:
    """Create an MP3 from text using OpenAI TTS. Truncates to keep it snappy on Cloud."""
    safe = text.strip()
    if len(safe) > max_chars:
        if log: log(f"TTS truncating from {len(safe):,} → {max_chars:,} chars for faster playback.")
        safe = safe[:max_chars] + "\n\n[...truncated for audio length...]"
    try:
        if log: log(f"TTS model={model}, voice={voice}")
        resp = client.audio.speech.create(model=model, voice=voice, input=safe, format="mp3")
        return resp.read()
    except Exception as e:
        if log: log(f"TTS fallback (tts-1): {e}")
        resp = client.audio.speech.create(model="tts-1", voice=voice, input=safe, format="mp3")
        return resp.read()

def timeline_preview(tl: List[Tuple[float, str]]) -> str:
    lines = []
    for start, text in tl[:40]:
        mm = int(start // 60); ss = int(start % 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {text}")
    if len(tl) > 40:
        lines.append("… (truncated)")
    return "\n".join(lines)

# =========================
# UI – one-click flow
# =========================
with st.sidebar:
    st.subheader("Summary Settings")
    tone = st.selectbox(
        "Tone",
        ["clear and professional", "friendly and concise", "analytical and direct", "executive brief"],
        index=2
    )
    target_minutes = st.slider("Target read length (minutes)", 5, 25, 12, 1)
    include_ts = st.checkbox("Include timestamps when possible", value=True)
    extra_focus = st.text_input("Optional focus areas (comma-separated)", value="regulatory risk, compute constraints")
    model_choice = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    tts_voice = st.selectbox("TTS Voice", ["alloy", "verse", "amber", "sage"], index=0)

# Single input + single button = single action
url = st.text_input("Paste podcast / episode URL (YouTube best for auto-transcript):").strip()
manual = st.text_area("Or paste transcript manually (auto-used if URL fails):", height=160)

go = st.button("Summarize & Play ▶️")

# Simple in-app logger
if "logs" not in st.session_state:
    st.session_state.logs = []
def log(msg: str):
    st.session_state.logs.append(msg)

# ============ One-click handler ============
if go:
    st.session_state.logs = []  # reset
    transcript_text: Optional[str] = None
    timeline: Optional[List[Tuple[float, str]]] = None

    with st.status("Working…", expanded=True) as status:
        # 1) Get transcript: URL → YT → manual
        if url:
            vid = extract_youtube_id(url)
            if vid:
                log(f"Detected YouTube ID: {vid}")
                try:
                    transcript_text, timeline = fetch_youtube_transcript(vid)
                    log(f"Fetched transcript: {len(transcript_text):,} characters.")
                except (TranscriptsDisabled, NoTranscriptFound):
                    log("No transcript available (captions disabled/unavailable).")
                except CouldNotRetrieveTranscript as e:
                    log(f"Transcript fetch returned empty/blocked: {e}")
                except Exception as e:
                    log(f"Unexpected YT fetch error: {type(e).__name__}: {e}")
            else:
                log("URL is not YouTube; auto-fetch not supported yet. Will use manual transcript if provided.")
        else:
            log("No URL provided; will use manual transcript if provided.")

        if not transcript_text:
            if manual.strip():
                transcript_text = manual.strip()
                log(f"Using manual transcript: {len(transcript_text):,} characters.")
            else:
                status.update(label="Need a transcript", state="error")
                st.error("Could not obtain a transcript. Paste one in the text area and click again.")
                st.stop()

        # 2) Build persona + summarize
        persona = build_persona_prompt(
            style=tone,
            target_minutes=target_minutes,
            include_timestamps=include_ts,
            extra_focus=[s.strip() for s in extra_focus.split(",")] if extra_focus else []
        )
        log("Persona prompt constructed.")
        try:
            summary_md = summarize_transcript(
                transcript=transcript_text,
                persona_prompt=persona,
                model=model_choice,
                temperature=temperature,
                log=log
            )
            log("Final merged summary ready.")
        except Exception as e:
            status.update(label="Summarization failed", state="error")
            st.error(f"Summarization failed: {type(e).__name__}: {e}")
            st.stop()

        # 3) Render summary
        st.success("Summary")
        st.markdown(summary_md)

        # 4) Auto-generate audio summary
        try:
            log("Generating audio summary (TTS)…")
            audio_bytes = tts_from_text(summary_md, voice=tts_voice, log=log)
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                "Download audio summary (.mp3)",
                data=audio_bytes,
                file_name="roadscout_summary.mp3",
                mime="audio/mpeg",
            )
            log("Audio ready.")
            status.update(label="Done", state="complete")
        except Exception as e:
            log(f"TTS failed: {type(e).__name__}: {e}")
            status.update(label="Summary ready (audio failed)", state="warning")
            st.warning(f"TTS failed: {type(e).__name__}: {e}")

# Diagnostics
with st.expander("Diagnostics (click to view logs)"):
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs))
    else:
        st.write("No logs yet. Paste a URL or transcript and click **Summarize & Play ▶️**.")

# Footer
st.caption("Notes: YouTube sometimes blocks transcripts (rate limits/region/captions off). For non-YouTube podcasts, paste transcript for now.")
