import os
import re
import io
import zipfile
from datetime import datetime
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# App Config + '70s Theme
# =========================
st.set_page_config(page_title="RoadScout: Podcast Summarizer", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Shrikhand&family=Kalam:wght@400;700&display=swap');

    :root{
      --rs-yellow:#FFD166;
      --rs-orange:#F25C05;
      --rs-red:#EF476F;
      --rs-teal:#06D6A0;
      --rs-deep:#073B4C;
      --rs-cream:#FFF7E6;
    }

    /* Groovy gradient background */
    .stApp {
      background: radial-gradient(circle at 15% 20%, var(--rs-yellow), transparent 40%),
                  radial-gradient(circle at 85% 30%, var(--rs-orange), transparent 45%),
                  radial-gradient(circle at 30% 80%, var(--rs-teal), transparent 45%),
                  linear-gradient(120deg, #fff, var(--rs-cream));
    }

    h1, .stMarkdown h1 { font-family: 'Shrikhand', cursive; color: var(--rs-deep); letter-spacing: 1px; }
    .stSidebar h2, .stSidebar h3 { font-family: 'Kalam', cursive; }

    /* Cards */
    .rs-card {
      background: #ffffffd9;
      border: 3px solid var(--rs-deep);
      border-radius: 18px;
      padding: 1rem 1.1rem;
      box-shadow: 6px 6px 0 var(--rs-deep);
      margin: 0.6rem 0 1rem 0;
    }
    .rs-title {
      font-family: 'Kalam', cursive;
      font-weight: 700;
      font-size: 1.05rem;
      color: var(--rs-deep);
      display: inline-flex;
      gap: .5rem;
      align-items: center;
    }
    .rs-chip { display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.8rem; margin-left:.25rem; }
    .rs-chip.yellow{ background:var(--rs-yellow); }
    .rs-chip.orange{ background:var(--rs-orange); color:white;}
    .rs-chip.teal{ background:var(--rs-teal); }
    .rs-section { border-top: 2px dashed var(--rs-deep); margin-top: .75rem; padding-top: .75rem;}
    .stButton>button {font-size: 1.05rem; padding: 0.6rem 1rem; border-radius: 14px; border:2px solid var(--rs-deep); box-shadow: 4px 4px 0 var(--rs-deep);}
    .stTextInput>div>div>input, .stTextArea textarea {font-size: 1rem;}
    .element-container:has(audio){ background:#fff; border-radius:12px; padding:.3rem .5rem; border:1px solid #ddd;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>🎧 RoadScout</h1>", unsafe_allow_html=True)
st.caption("Paste a podcast/YouTube URL (or transcript), tap once, and get a persona-driven text + audio summary.")

# Inline JS for completion ding
st.markdown(
    """
    <script>
    function rs_playDing(){
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = 'triangle';
      o.frequency.value = 880; // A5
      o.connect(g); g.connect(ctx.destination);
      g.gain.setValueAtTime(0.0001, ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.45);
      o.start(); o.stop(ctx.currentTime + 0.5);
    }
    </script>
    """,
    unsafe_allow_html=True,
)

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
        raise CouldNotRetrieveTranscript("Transcript fetch returned empty content.")
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

def chunk_text(text: str, max_chars: int = 24000) -> List[str]:
    """
    Larger chunks reduce API calls. Hard-splits ensure we never ship a giant block.
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
                         model_merge_and_fit: str = "gpt-4.1",
                         model_per_chunk: Optional[str] = None,
                         temperature: float = 0.2,
                         log=None) -> str:
    """
    Summarize each chunk with model_per_chunk (defaults to merge model if None),
    then merge+fit with model_merge_and_fit.
    """
    chunks = chunk_text(transcript, max_chars=24000)
    if log: log(f"Transcript length: {len(transcript):,} chars → {len(chunks)} chunk(s).")
    intermediates: List[str] = []
    per_chunk_model = model_per_chunk or model_merge_and_fit

    for i, ch in enumerate(chunks, 1):
        if log: log(f"Summarizing chunk {i}/{len(chunks)} ({len(ch):,} chars) with {per_chunk_model}…")
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": f"TRANSCRIPT CHUNK {i}/{len(chunks)}:\n\n{ch}\n\nProvide a compact intermediate summary (bulleted where helpful)."}
        ]
        resp = client.chat.completions.create(
            model=per_chunk_model,
            messages=messages,
            temperature=temperature,
        )
        intermediates.append(resp.choices[0].message.content.strip())

    if log: log(f"Merging {len(intermediates)} summaries with {model_merge_and_fit}…")
    merge_messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": "Merge the following intermediate summaries into a single cohesive final report. Remove redundancies, keep structure, and ensure chronology and lenses are clear:\n\n" + "\n\n---\n\n".join(intermediates)}
    ]
    final_resp = client.chat.completions.create(
        model=model_merge_and_fit,
        messages=merge_messages,
        temperature=temperature,
    )
    return final_resp.choices[0].message.content.strip()

# ==== Length control ====
def estimate_words_from_minutes(minutes: int, wpm: int = 170) -> int:
    return max(200, int(minutes * wpm))

def fit_summary_to_length(raw_md: str,
                          persona_prompt: str,
                          target_minutes: int,
                          model: str,
                          temperature: float,
                          log=None) -> str:
    target_words = estimate_words_from_minutes(target_minutes)
    if log: log(f"Fitting summary to ~{target_words} words (≈{target_minutes} min @170 wpm).")
    soft_token_cap = int(target_words * 1.4) + 200
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

# ---------- FULL-LENGTH AUDIO (multipart + parallel) ----------
def split_for_tts(text: str, part_chars: int = 5500) -> List[str]:
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

def tts_multipart_parallel(summary_md: str, voice: str, max_workers: int = 3, log=None) -> List[Tuple[int, bytes]]:
    parts_text = split_for_tts(summary_md, part_chars=5500)
    if log: log(f"TTS will generate {len(parts_text)} part(s) in parallel ({max_workers} workers).")
    results: List[Tuple[int, bytes]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_tts_bytes, pt, voice, log): idx for idx, pt in enumerate(parts_text, 1)}
        for fut in as_completed(futures):
            idx = futures[fut]
            audio = fut.result()
            results.append((idx, audio))
    results.sort(key=lambda x: x[0])
    return results


# =========================
# Sidebar – controls
# =========================
with st.sidebar:
    st.markdown('<div class="rs-card"><span class="rs-title">🎛️ Controls <span class="rs-chip teal">New</span></span>', unsafe_allow_html=True)
    tone = st.selectbox(
        "Tone",
        ["clear and professional", "friendly and concise", "analytical and direct", "executive brief"],
        index=2
    )
    target_minutes = st.slider("Target read length (minutes)", 5, 25, 12, 1)
    include_ts = st.checkbox("Include timestamps when possible", value=True)
    extra_focus = st.text_input("Optional focus areas (comma-separated)", value="regulatory risk, compute constraints")
    model_choice = st.selectbox("Merge/Fit Model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"], index=0)
    speed_mode = st.checkbox("Speed mode (use gpt-4o-mini for per-chunk)", value=True)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    tts_voice = st.selectbox("TTS Voice", ["alloy", "verse", "amber", "sage"], index=0)
    play_ding = st.checkbox("Play a ding when finished", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main inputs
st.markdown('<div class="rs-card"><span class="rs-title">🔗 Input</span><span class="rs-chip yellow">One-click</span><div class="rs-section"></div>', unsafe_allow_html=True)
url = st.text_input("Paste podcast / episode URL (YouTube works best for auto-transcript):").strip()
manual = st.text_area("Or paste transcript manually (auto-used if URL fails):", height=130)
st.markdown('</div>', unsafe_allow_html=True)

go = st.button("Summarize & Play ▶️", use_container_width=True)

# Logger
if "logs" not in st.session_state:
    st.session_state.logs = []
def log(msg: str):
    st.session_state.logs.append(msg)

# ============ One-click handler ============
if go:
    st.session_state.logs = []
    transcript_text: Optional[str] = None
    timeline: Optional[List[Tuple[float, str]]] = None
    episode_title: Optional[str] = None

    with st.status("Working…", expanded=True) as status:
        # 1) Transcript
        if url:
            vid = extract_youtube_id(url)
            if vid:
                log(f"Detected YouTube ID: {vid}")
                try:
                    transcript_text, timeline = fetch_youtube_transcript(vid)
                    log(f"[yt-transcript-api] OK – {len(transcript_text):,} chars.")
                except (TranscriptsDisabled, NoTranscriptFound) as e:
                    log(f"[yt-transcript-api] No transcript: {e}")
                except CouldNotRetrieveTranscript as e:
                    log(f"[yt-transcript-api] Empty/blocked: {e}")
                except Exception as e:
                    log(f"[yt-transcript-api] Unexpected: {type(e).__name__}: {e}")
                if not transcript_text:
                    try:
                        transcript_text, episode_title, timeline = fetch_info_and_captions_via_ytdlp(url)
                        log(f"[yt-dlp] captions OK – {len(transcript_text):,} chars.")
                        if episode_title: log(f"Title: {episode_title}")
                    except Exception as e:
                        log(f"[yt-dlp] fallback failed: {type(e).__name__}: {e}")
            else:
                log("URL is not YouTube; manual transcript only for now.")
        if not transcript_text:
            if manual.strip():
                transcript_text = manual.strip()
                log(f"Using manual transcript: {len(transcript_text):,} chars.")
            else:
                status.update(label="Need a transcript", state="error")
                st.error("Could not obtain a transcript. Paste one and click again.")
                st.stop()

        # 2) Persona + summarize (speed mode uses mini for chunks)
        persona = build_persona_prompt(
            style=tone,
            target_minutes=target_minutes,
            include_timestamps=include_ts,
            extra_focus=[s.strip() for s in extra_focus.split(",")] if extra_focus else []
        )
        log("Persona prompt ready.")
        try:
            per_chunk_model = "gpt-4o-mini" if speed_mode else None
            summary_md = summarize_transcript(
                transcript=transcript_text,
                persona_prompt=persona,
                model_merge_and_fit=model_choice,
                model_per_chunk=per_chunk_model,
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

        # 3) Render summary & download
        st.markdown('<div class="rs-card"><span class="rs-title">📝 End of Report</span><div class="rs-section"></div>', unsafe_allow_html=True)
        st.markdown(summary_md)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = slugify(episode_title or "roadscout-summary")
        md_bytes = io.BytesIO(summary_md.encode("utf-8"))
        st.download_button("Download summary (.md)", md_bytes, file_name=f"{base}_{ts}.md", mime="text/markdown")
        st.markdown('</div>', unsafe_allow_html=True)

        # 4) Audio summary (parallel parts)
        try:
            log("Generating audio summary (parallel parts)…")
            parts = tts_multipart_parallel(summary_md, voice=tts_voice, max_workers=3, log=log)

            for i, audio_bytes in parts:
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button(
                    f"Download Part {i} (.mp3)",
                    data=audio_bytes,
                    file_name=f"{base}_{ts}_part{i}.mp3",
                    mime="audio/mpeg",
                    key=f"dl_part_{i}"
                )

            zip_mem = io.BytesIO()
            with zipfile.ZipFile(zip_mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i, audio_bytes in parts:
                    zf.writestr(f"{base}_{ts}_part{i}.mp3", audio_bytes)
            zip_mem.seek(0)
            st.download_button("Download all parts (.zip)", zip_mem.read(), file_name=f"{base}_{ts}.zip", mime="application/zip")

            status.update(label="Done", state="complete")
            if play_ding:
                st.markdown("<script>rs_playDing()</script>", unsafe_allow_html=True)
            log("Audio parts ready.")
        except Exception as e:
            status.update(label="Summary ready (audio failed)", state="complete")
            st.warning(f"TTS failed: {type(e).__name__}: {e}")
            if play_ding:
                st.markdown("<script>rs_playDing()</script>", unsafe_allow_html=True)
            log(f"TTS failed: {type(e).__name__}: {e}")

# Diagnostics
with st.expander("Diagnostics (click to view logs)"):
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs))
    else:
        st.write("No logs yet. Paste a URL or transcript and tap **Summarize & Play ▶️**.")

st.caption("Notes: YouTube sometimes blocks transcripts (rate limits/region/captions off). For non-YouTube podcasts, paste transcript for now.")
