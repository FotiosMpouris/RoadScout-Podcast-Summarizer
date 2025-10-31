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

# ---- Optional: YouTube transcript fetcher ----
# Lightweight and Streamlit-Cloud friendly (no ffmpeg)
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        CouldNotRetrieveTranscript,
    )
except Exception:
    # We'll degrade gracefully if this isn't available
    YouTubeTranscriptApi = None

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

# Expect OPENAI_API_KEY in Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Add `OPENAI_API_KEY` to Streamlit Secrets.")
    st.stop()
st.caption("🔐 Secrets OK") if OPENAI_API_KEY else st.error("No OPENAI_API_KEY detected")
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
    Returns (joined_text, [(start_seconds, text), ...])
    If no transcript, raises a handled exception.
    """
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api is not installed.")
    languages = languages or ["en", "en-US", "en-GB", "auto"]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer English; fall back to generated
        candidates = []
        for code in languages:
            try:
                candidates.append(transcript_list.find_transcript([code]))
            except Exception:
                continue
        if not candidates:
            # Fallback: the first transcript available
            candidates = [next(iter(transcript_list), None)]
        chosen = next((c for c in candidates if c is not None), None)
        if not chosen:
            raise NoTranscriptFound("No transcript candidate found.")
        data = chosen.fetch()
        # Build timeline text
        timeline = [(item["start"], item["text"]) for item in data]
        joined = " ".join(seg for _, seg in timeline)
        return joined, timeline
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as e:
        raise e

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
                         temperature: float = 0.2) -> str:
    """
    Sends chunked transcript to the model with a system+user flow:
    - system: persona_prompt
    - user: chunk i
    Then merges with a final pass.
    """
    # 1) Per-chunk intermediate summaries
    chunks = chunk_text(transcript, max_chars=12000)
    intermediates: List[str] = []

    for i, ch in enumerate(chunks, 1):
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

    # 2) Merge pass
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

def timeline_with_timestamps(timeline: List[Tuple[float, str]]) -> str:
    """Utility for showing a simple preview of fetched transcript with timestamps."""
    preview_lines = []
    for start, text in timeline[:40]:  # limit preview
        mm = int(start // 60)
        ss = int(start % 60)
        preview_lines.append(f"[{mm:02d}:{ss:02d}] {text}")
    if len(timeline) > 40:
        preview_lines.append("... (truncated preview)")
    return "\n".join(preview_lines)

# =========================
# UI
# =========================
st.title("🎧 RoadScout: Podcast Summarizer")
st.caption("Paste a podcast or YouTube URL, fetch transcript (when available), and get a persona-driven summary.")

with st.sidebar:
    st.subheader("Summary Settings")
    tone = st.selectbox(
        "Tone",
        ["clear and professional", "friendly and concise", "analytical and direct", "executive brief"],
        index=2
    )
    target_minutes = st.slider("Target read length (minutes)", min_value=5, max_value=25, value=12, step=1)
    include_ts = st.checkbox("Include timestamps when possible", value=True)
    extra_focus = st.text_input("Optional: extra focus areas (comma-separated)", value="regulatory risk, compute constraints")
    model_choice = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)

url = st.text_input("Paste podcast / episode URL (YouTube works best for auto-transcript):").strip()
st.write("— or —")
manual = st.text_area("Paste transcript manually (if no transcript can be fetched):", height=200)

colA, colB = st.columns([1, 1])

with colA:
    fetch_btn = st.button("Fetch Transcript (from URL)")
with colB:
    summarize_btn = st.button("Summarize")

transcript_text: Optional[str] = None
timeline: Optional[List[Tuple[float, str]]] = None

# Fetch transcript flow
if fetch_btn:
    if not url:
        st.warning("Please paste a URL first.")
    else:
        yt_id = extract_youtube_id(url)
        if yt_id:
            try:
                with st.spinner("Fetching YouTube transcript..."):
                    transcript_text, timeline = fetch_youtube_transcript(yt_id)
                st.success("Transcript fetched successfully.")
                with st.expander("Transcript Preview (first ~40 entries)"):
                    st.code(timeline_with_timestamps(timeline))
                with st.expander("Raw Transcript (joined)"):
                    st.write(transcript_text[:4000] + ("..." if len(transcript_text) > 4000 else ""))
            except Exception as e:
                st.error(f"Could not fetch YouTube transcript: {e}")
        else:
            st.info("Non-YouTube URLs are not auto-supported yet. Paste the transcript below and click Summarize.")

# Summarize flow
if summarize_btn:
    # Decide transcript source
    if not manual and not transcript_text:
        if url and extract_youtube_id(url):
            st.warning("Try 'Fetch Transcript' first, or paste the transcript manually.")
        else:
            st.warning("Please paste a transcript in the text area, then click Summarize.")
    else:
        effective_transcript = manual or transcript_text
        persona = build_persona_prompt(
            style=tone,
            target_minutes=target_minutes,
            include_timestamps=include_ts,
            extra_focus=[s.strip() for s in extra_focus.split(",")] if extra_focus else []
        )
        with st.spinner("Summarizing…"):
            try:
                summary_md = summarize_transcript(
                    transcript=effective_transcript,
                    persona_prompt=persona,
                    model=model_choice,
                    temperature=temperature
                )
                st.success("Summary ready!")
                st.markdown(summary_md)

                # Download button
                md_bytes = io.BytesIO(summary_md.encode("utf-8"))
                st.download_button(
                    label="Download summary (.md)",
                    data=md_bytes,
                    file_name="roadscout_summary.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Summarization failed: {e}")

# Footer note
st.caption("Tip: For non-YouTube podcasts, paste the transcript manually for now. Auto-fetch for RSS/audio is on the roadmap.")
