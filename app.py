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

# ---- YouTube transcript fetchers ----
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        CouldNotRetrieveTranscript,
    )
except Exception:
    YouTubeTranscriptApi = None  # degrade gracefully

import requests
from requests.exceptions import RequestException

# yt-dlp for fallback caption discovery
try:
    import yt_dlp
except Exception:
    yt_dlp = None  # degrade gracefully

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

# ---- Primary fetch using youtube-transcript-api ----
def fetch_youtube_transcript(video_id: str, languages: Optional[List[str]] = None) -> Tuple[str, List[Tuple[float, str]]]:
    """
    Robust transcript fetcher with youtube-transcript-api:
    - Try official captions first
    - Then try auto-generated captions
    Returns (joined_text, [(start_seconds, text), ...])
    """
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api is not installed on this deployment.")
    languages = languages or ["en", "en-US", "en-GB"]

    try:
        data = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except Exception:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        chosen = None
        for code in (languages + ["en"]):
            try:
                chosen = transcripts.find_transcript([code])
                break
            except Exception:
                continue
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
        raise CouldNotRetrieveTranscript(
            "Transcript fetch returned empty content (YouTube may be blocking or captions are disabled)."
        )
    return joined, timeline

# ---- Fallback fetch using yt-dlp + VTT parsing ----
VTT_TS = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}\.\d{3})\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}"
)

def _parse_vtt(vtt_text: str) -> List[Tuple[float, str]]:
    """
    Very small WebVTT parser (enough for YouTube subs).
    Returns [(start_seconds, text)]
    """
    lines = vtt_text.splitlines()
    result: List[Tuple[float, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = VTT_TS.match(line)
        if m:
            # collect caption text until blank line
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip():
                # strip simple VTT tags like <c> or <i>
                text_lines.append(re.sub(r"<[^>]+>", "", lines[i]).strip())
                i += 1
            start = int(m.group("h")) * 3600 + int(m.group("m")) * 60 + float(m.group("s"))
            text = " ".join(t for t in text_lines if t)
            if text:
                result.append((start, text))
        else:
            i += 1
    return result

def fetch_youtube_transcript_via_ytdlp(url: str) -> Tuple[str, List[Tuple[float, str]]]:
    """
    Uses yt-dlp to locate captions (official or auto) and downloads VTT to parse.
    """
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed on this deployment.")
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    # Prefer official English subs, then auto English, then any English
    def pick_track(container: dict, key: str) -> Optional[str]:
        if not container or key not in container:
            return None
        # Try exact 'en' first; else any 'en-*'
        if "en" in container[key]:
            tracks = container[key]["en"]
        else:
            lang = next((k for k in container[key] if k.startswith("en")), None)
            if not lang:
                return None
            tracks = container[key][lang]
        # Prefer VTT
        vtt = next((t for t in tracks if t.get("ext") == "vtt"), None)
        if vtt:
            return vtt.get("url")
        return tracks[0].get("url") if tracks else None

    vtt_url = (pick_track(info, "subtitles")
               or pick_track(info, "automatic_captions"))
    if not vtt_url:
        raise NoTranscriptFound("No captions found via yt-dlp.")
    try:
        r = requests.get(vtt_url, timeout=15)
        r.raise_for_status()
        timeline = _parse_vtt(r.text)
        joined = " ".join(seg for _, seg in timeline).strip()
        if not joined:
            raise CouldNotRetrieveTranscript("Empty VTT content.")
        return joined, timeline
    except RequestException as e:
        raise CouldNotRetrieveTranscript(f"Failed to download captions: {e}")

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
    """Chunked summarize + merge."""
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
        # 1) Get transcript: URL → YT (primary) → yt-dlp fallback → manual
        if url:
            vid = extract_youtube_id(url)
            if vid:
                log(f"Detected YouTube ID: {vid}")
                # Primary: youtube-transcript-api
                try:
                    transcript_text, timeline = fetch_youtube_transcript(vid)
                    log(f"[yt-transcript-api] OK – {len(transcript_text):,} chars.")
                except (TranscriptsDisabled, NoTranscriptFound) as e:
                    log(f"[yt-transcript-api] No transcript: {e}")
                except CouldNotRetrieveTranscript as e:
                    log(f"[yt-transcript-api] Empty/blocked: {e}")
                except Exception as e:
                    log(f"[yt-transcript-api] Unexpected: {type(e).__name__}: {e}")

                # Fallback: yt-dlp captions
                if not transcript_text:
                    try:
                        if yt_dlp is None:
                            raise RuntimeError("yt-dlp not installed")
                        transcript_text, timeline = fetch_youtube_transcript_via_ytdlp(url)
                        log(f"[yt-dlp] captions OK – {len(transcript_text):,} chars.")
                    except Exception as e:
                        log(f"[yt-dlp] fallback failed: {type(e).__name__}: {e}")
            else:
                log("URL is not YouTube; auto-fetch not supported yet. Will use manual transcript if provided.")
        else:
            log("No URL provided; will use manual transcript if provided.")

        if not transcript_text:
            if manual.strip():
                transcript_text = manual.strip()
                log(f"Using manual transcript: {len(transcript_text):,} chars.")
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
