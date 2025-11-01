import os
import re
import io
import zipfile
from datetime import datetime
from typing import List, Optional, Tuple

import streamlit as st

# ---- OpenAI client (official SDK) ----
try:
    from openai import OpenAI
except ImportError:
    st.error("Missing OpenAI SDK. Make sure 'openai' is in requirements.txt.")
    st.stop()

# ---- YouTube transcript (primary) ----
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        CouldNotRetrieveTranscript,
    )
except Exception:
    YouTubeTranscriptApi = None  # degrade gracefully

# ---- Fallback caption discovery/downloader ----
import requests
from requests.exceptions import RequestException

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
# App Config (mobile-friendly)
# =========================
st.set_page_config(page_title="RoadScout: Podcast Summarizer", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
    /* Slightly larger controls for mobile */
    .stButton>button {font-size: 1.05rem; padding: 0.6rem 1rem;}
    .stTextInput>div>div>input {font-size: 1rem;}
    .stTextArea textarea {font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎧 RoadScout: Podcast Summarizer")
st.caption("Paste a podcast/YouTube URL (or transcript), tap once, and get a persona-driven text + audio summary.")

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

def slugify(name: str, max_len: int = 80) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[^A-Za-z0-9 _-]+", "", name)
    name = name.replace(" ", "-")
    return name[:max_len] or "podcast"

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
    lines = vtt_text.splitlines()
    result: List[Tuple[float, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = VTT_TS.match(line)
        if m:
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip():
                text_lines.append(re.sub(r"<[^>]+>", "", lines[i]).strip())
                i += 1
            start = int(m.group("h")) * 3600 + int(m.group("m")) * 60 + float(m.group("s"))
            text = " ".join(t for t in text_lines if t)
            if text:
                result.append((start, text))
        else:
            i += 1
    return result

def fetch_info_and_captions_via_ytdlp(url: str) -> Tuple[str, Optional[str], List[Tuple[float, str]]]:
    """
    Uses yt-dlp to get title and captions (official/auto) as VTT.
    Returns (joined_text, title, timeline)
    """
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed on this deployment.")
    ydl_opts = {"quiet": True, "skip_download": True, "writesubtitles": True, "writeautomaticsub": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title")

    def pick_track(container: dict, key: str) -> Optional[str]:
        if not container or key not in container:
            return None
        if "en" in container[key]:
            tracks = container[key]["en"]
        else:
            lang = next((k for k in container[key] if k.startswith("en")), None)
            if not lang:
                return None
            tracks = container[key][lang]
        vtt = next((t for t in tracks if t.get("ext") == "vtt"), None)
        if vtt:
            return vtt.get("url")
        return tracks[0].get("url") if tracks else None

    vtt_url = (pick_track(info, "subtitles") or pick_track(info, "automatic_captions"))
    if not vtt_url:
        raise NoTranscriptFound("No captions found via yt-dlp.")
    r = requests.get(vtt_url, timeout=15)
    r.raise_for_status()
    timeline = _parse_vtt(r.text)
    joined = " ".join(seg for _, seg in timeline).strip()
    if not joined:
        raise CouldNotRetrieveTranscript("Empty VTT content.")
    return joined, title, timeline

def chunk_text(text: str, max_chars: int = 20000) -> List[str]:
    """
    Split transcript into ~max_chars chunks.
    Prefers paragraph breaks, but hard-splits long paragraphs so we never
    end up with a single huge chunk.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paras = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    buf = ""

    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if buf:
                chunks.append(buf); buf = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i + max_chars])
            continue
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= max_chars:
            buf = f"{buf}\n\n{para}"
        else:
            chunks.append(buf); buf = para

    if buf:
        chunks.append(buf)
    return chunks

def summarize_transcript(transcript: str,
                         persona_prompt: str,
                         model: str = "gpt-4.1",
                         temperature: float = 0.2,
                         log=None) -> str:
    """Chunked summarize + merge."""
    chunks = chunk_text(transcript, max_chars=20000)
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

# ==== Length control ====
def estimate_words_from_minutes(minutes: int, wpm: int = 170) -> int:
    """Target words for the final summary (aloud)."""
    return max(200, int(minutes * wpm))

def fit_summary_to_length(raw_md: str,
                          persona_prompt: str,
                          target_minutes: int,
                          model: str,
                          temperature: float,
                          log=None) -> str:
    """Final 'fit' pass: compress/expand to ~N words while keeping structure."""
    target_words = estimate_words_from_minutes(target_minutes)
    if log: log(f"Fitting summary to ~{target_words} words (≈{target_minutes} min at ~170 wpm).")
    soft_token_cap = int(target_words * 1.4) + 200  # generous cap

    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content":
            f"""Rewrite the following summary to approximately {target_words} words (±10%) while retaining the same section structure and key points.
Keep it concise but substantive; do not remove mandatory headings.

SUMMARY TO ADJUST:
{raw_md}
"""}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=soft_token_cap
    )
    return resp.choices[0].message.content.strip()

# ---------- FULL-LENGTH AUDIO (multipart) ----------
def split_for_tts(text: str, part_chars: int = 4500) -> List[str]:
    """Split on paragraph boundaries when possible; hard-split if needed."""
    text = text.strip()
    if len(text) <= part_chars:
        return [text]
    parts: List[str] = []
    buf = ""
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if len(para) > part_chars:
            if buf:
                parts.append(buf); buf = ""
            for i in range(0, len(para), part_chars):
                parts.append(para[i:i+part_chars])
            continue
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= part_chars:
            buf = f"{buf}\n\n{para}"
        else:
            parts.append(buf); buf = para
    if buf:
        parts.append(buf)
    return parts

def _tts_bytes(text: str, voice: str, log=None) -> bytes:
    """
    TTS wrapper compatible with your SDK variant (no 'format' kwarg).
    Tries streaming first, then non-streaming, then tts-1.
    """
    try:
        if log: log("TTS streaming (primary)")
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
        ) as resp:
            return resp.read()
    except Exception as e:
        if log: log(f"TTS streaming failed: {e}")
    try:
        if log: log("TTS create() (secondary)")
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
        )
        return resp.read()
    except Exception as e:
        if log: log(f"TTS create() failed: {e}")
    if log: log("TTS fallback to tts-1")
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
    ) as resp:
        return resp.read()

def tts_multipart(summary_md: str, voice: str, log=None) -> List[Tuple[str, bytes]]:
    """Produce multiple MP3 parts and return [(filename, bytes), ...]."""
    parts_text = split_for_tts(summary_md, part_chars=4500)
    if log: log(f"TTS will generate {len(parts_text)} part(s).")
    results: List[Tuple[str, bytes]] = []
    for idx, pt in enumerate(parts_text, 1):
        if log: log(f"TTS Part {idx} ({len(pt)} chars)")
        audio = _tts_bytes(pt, voice=voice, log=log)
        results.append((f"roadscout_summary_part{idx}.mp3", audio))
    return results

def zip_parts(parts: List[Tuple[str, bytes]]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, b in parts:
            zf.writestr(fname, b)
    mem.seek(0)
    return mem.read()


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
    # Default to gpt-4.1 as requested
    model_choice = st.selectbox("Model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    tts_voice = st.selectbox("TTS Voice", ["alloy", "verse", "amber", "sage"], index=0)

# Single input + single button
url = st.text_input("Paste podcast / episode URL (YouTube best for auto-transcript):").strip()
manual = st.text_area("Or paste transcript manually (auto-used if URL fails):", height=140)

# Bigger, mobile-friendly primary action
go = st.button("Summarize & Play ▶️", use_container_width=True)

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
    episode_title: Optional[str] = None

    with st.status("Working…", expanded=True) as status:
        # 1) Get transcript: URL → YT API → yt-dlp fallback → manual
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

                # Fallback: yt-dlp captions + title
                if not transcript_text:
                    try:
                        if yt_dlp is None:
                            raise RuntimeError("yt-dlp not installed")
                        transcript_text, episode_title, timeline = fetch_info_and_captions_via_ytdlp(url)
                        log(f"[yt-dlp] captions OK – {len(transcript_text):,} chars.")
                        if episode_title:
                            log(f"Title: {episode_title}")
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
            log("Merging done. Running length fit…")
            summary_md = fit_summary_to_length(
                raw_md=summary_md,
                persona_prompt=persona,
                target_minutes=target_minutes,
                model=model_choice,
                temperature=temperature,
                log=log
            )
            log("Final summary ready.")
        except Exception as e:
            status.update(label="Summarization failed", state="error")
            st.error(f"Summarization failed: {type(e).__name__}: {e}")
            st.stop()

        # 3) Render summary + download (title-aware filename)
        st.success("Summary")
        st.markdown(summary_md)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = slugify(episode_title or "roadscout-summary")
        md_bytes = io.BytesIO(summary_md.encode("utf-8"))
        st.download_button(
            label="Download summary (.md)",
            data=md_bytes,
            file_name=f"{base}_{ts}.md",
            mime="text/markdown"
        )

        # 4) Auto-generate audio summary (multipart, no truncation)
        try:
            log("Generating audio summary parts…")
            parts = tts_multipart(summary_md, voice=tts_voice, log=log)

            # Show inline players + per-part downloads
            for i, (fname, audio_bytes) in enumerate(parts, 1):
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button(
                    f"Download Part {i} (.mp3)",
                    data=audio_bytes,
                    file_name=f"{base}_{ts}_part{i}.mp3",
                    mime="audio/mpeg",
                    key=f"dl_part_{i}"
                )

            # One-click ZIP for car rides
            zip_mem = io.BytesIO()
            with zipfile.ZipFile(zip_mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i, (fname, audio_bytes) in enumerate(parts, 1):
                    zf.writestr(f"{base}_{ts}_part{i}.mp3", audio_bytes)
            zip_mem.seek(0)
            st.download_button(
                "Download all parts (.zip)",
                data=zip_mem.read(),
                file_name=f"{base}_{ts}.zip",
                mime="application/zip"
            )

            log("Audio parts ready.")
            status.update(label="Done", state="complete")
        except Exception as e:
            log(f"TTS failed: {type(e).__name__}: {e}")
            status.update(label="Summary ready (audio failed)", state="complete")
            st.warning(f"TTS failed: {type(e).__name__}: {e}")

# Diagnostics (collapsed by default for mobile sanity)
with st.expander("Diagnostics (click to view logs)"):
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs))
    else:
        st.write("No logs yet. Paste a URL or transcript and tap **Summarize & Play ▶️**.")

# Footer
st.caption("Notes: YouTube sometimes blocks transcripts (rate limits/region/captions off). For non-YouTube podcasts, paste transcript for now.")
