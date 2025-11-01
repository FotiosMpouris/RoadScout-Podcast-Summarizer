# audio_utils.py
import os
import re
import tempfile
import subprocess
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Public API used by app.py ----
def tts_to_single_mp3(
    client,
    text: str,
    voice: str = "alloy",
    model: str = "tts-1",
    max_workers: int = 3,
    max_chars_per_chunk: int = 3800,  # <= 4096 safe for TTS input
) -> bytes:
    """
    Splits 'text' into chunks, synthesizes each chunk to MP3 using OpenAI TTS,
    then concatenates all MP3s into a single MP3 using ffmpeg's concat demuxer.
    Returns MP3 bytes.
    """
    chunks = _split_text(text, max_chars_per_chunk)
    if not chunks:
        raise ValueError("No text to synthesize.")

    # 1) Synthesize chunks in parallel to temporary MP3 files
    tmp_dir = tempfile.mkdtemp(prefix="rs_tts_")
    mp3_paths: List[str] = []

    def _synth(i_and_chunk):
        i, ch = i_and_chunk
        out_path = os.path.join(tmp_dir, f"part_{i:04d}.mp3")
        # Use streaming API to write MP3 directly to disk
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=ch,
        ) as resp:
            resp.stream_to_file(out_path)
        return out_path

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_synth, (i, ch)) for i, ch in enumerate(chunks)]
        for f in as_completed(futures):
            mp3_paths.append(f.result())

    # Ensure files are ordered 0..N
    mp3_paths.sort()

    # 2) Concatenate with ffmpeg (no re-encoding, just stream copy)
    list_txt = os.path.join(tmp_dir, "concat_list.txt")
    with open(list_txt, "w", encoding="utf-8") as f:
        for p in mp3_paths:
            # ffmpeg concat requires exact "file 'path'"
            f.write(f"file '{p}'\n")

    final_mp3 = os.path.join(tmp_dir, "final.mp3")
    _run_ffmpeg_concat(list_txt, final_mp3)

    # 3) Read bytes, cleanup temp dir
    with open(final_mp3, "rb") as f:
        data = f.read()

    _cleanup_dir(tmp_dir)
    return data


# ---- Helpers ----
def _split_text(text: str, max_chars: int) -> List[str]:
    """
    Split on double newlines where possible, otherwise hard-wrap.
    """
    text = (text or "").strip()
    if not text:
        return []

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
            # flush buffer first
            if buf:
                chunks.append(buf); buf = ""
            # hard split long paragraph
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i + max_chars])
        else:
            if not buf:
                buf = para
            elif len(buf) + 2 + len(para) <= max_chars:
                buf = f"{buf}\n\n{para}"
            else:
                chunks.append(buf); buf = para

    if buf:
        chunks.append(buf)

    return chunks


def _run_ffmpeg_concat(list_file: str, out_path: str) -> None:
    """
    Uses ffmpeg concat demuxer to join MP3 files losslessly.
    Requires 'ffmpeg' in PATH (Streamlit Cloud: install via packages.txt).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"ffmpeg concat failed: {proc.stderr.strip()}")


def _cleanup_dir(path: str) -> None:
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except OSError:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except OSError:
                    pass
        os.rmdir(path)
    except OSError:
        pass
