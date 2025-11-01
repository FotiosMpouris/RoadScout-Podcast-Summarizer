# audio_utils.py
import io
import re
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydub import AudioSegment

# OpenAI client is created in app.py and passed in
# We stay tool-agnostic here.


# ---------- light text normalization to improve pronunciation ----------
_ABBR_MAP = [
    (r"\bF(arenh(e|)ight)\b", "Fahrenheit"),
    (r"°\s*F\b", " degrees Fahrenheit"),
    (r"\b°F\b", " degrees Fahrenheit"),
    (r"\bCelsius\b", " Celsius"),
    (r"°\s*C\b", " degrees Celsius"),
    (r"\b°C\b", " degrees Celsius"),
    (r"\bHz\b", " hertz"),
    (r"\bkHz\b", " kilohertz"),
    (r"\bMHz\b", " megahertz"),
    (r"\bGHz\b", " gigahertz"),
    (r"\bi\.e\.\b", " that is"),
    (r"\be\.g\.\b", " for example"),
    (r"\bet al\.\b", " et al"),
]

def normalize_for_tts(text: str) -> str:
    out = text
    for pat, rep in _ABBR_MAP:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    # collapse triple newlines → doubles (helps pacing, avoids empty clips)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------- chunking ----------
def split_text(text: str, part_chars: int = 5200) -> List[str]:
    """
    Break text roughly on paragraph boundaries; hard split long paragraphs.
    5.2k chars is safe for TTS latencies while still minimizing part count.
    """
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
            # flush
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


# ---------- low-level OpenAI TTS (bytes) ----------
def _tts_bytes_openai(client, text: str, voice: str) -> bytes:
    """
    Try streaming gpt-4o-mini-tts first; fallback to tts-1.
    Returns raw MP3 bytes or raises.
    """
    # primary (streaming)
    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
        ) as resp:
            return resp.read()
    except Exception:
        pass
    # fallback
    try:
        resp = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        return resp.read()
    except Exception as e:
        raise e


# ---------- main: one-shot single MP3 builder ----------
def tts_to_single_mp3(
    client,
    text: str,
    voice: str = "alloy",
    max_workers: int = 3,
) -> bytes:
    """
    1) normalizes text
    2) splits into parts
    3) synthesizes parts in parallel (OpenAI TTS)
    4) concatenates in-memory with pydub to ONE MP3
    Returns final MP3 bytes.
    """
    clean = normalize_for_tts(text)
    parts = split_text(clean, part_chars=5200)

    # synthesize in parallel → [(idx, bytes)]
    results: List[Tuple[int, bytes]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_tts_bytes_openai, client, p, voice): i for i, p in enumerate(parts, 1)}
        for fut in as_completed(futs):
            idx = futs[fut]
            audio = fut.result()
            results.append((idx, audio))
    results.sort(key=lambda x: x[0])

    # stitch using pydub
    combined = AudioSegment.empty()
    for _, mp3_bytes in results:
        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        combined += seg

    out = io.BytesIO()
    combined.export(out, format="mp3")
    return out.getvalue()
