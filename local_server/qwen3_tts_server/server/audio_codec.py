"""音频格式编码 — PCM float32 → WAV/PCM/MP3/OPUS/FLAC/AAC"""
import io
import struct
import subprocess
import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000


def pcm_to_wav(pcm_float: np.ndarray) -> bytes:
    """float32 PCM → WAV bytes (16-bit PCM, 24kHz, mono)"""
    buf = io.BytesIO()
    sf.write(buf, pcm_float, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pcm_to_raw(pcm_float: np.ndarray) -> bytes:
    """float32 PCM → raw PCM int16 LE bytes"""
    pcm_int16 = (pcm_float * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm_int16.tobytes()


def pcm_to_flac(pcm_float: np.ndarray) -> bytes:
    """float32 PCM → FLAC bytes"""
    buf = io.BytesIO()
    sf.write(buf, pcm_float, SAMPLE_RATE, format="FLAC")
    return buf.getvalue()


def pcm_to_mp3(pcm_float: np.ndarray) -> bytes:
    """float32 PCM → MP3 via ffmpeg"""
    return _ffmpeg_encode(pcm_float, "mp3")


def pcm_to_opus(pcm_float: np.ndarray) -> bytes:
    """float32 PCM → OGG/Opus via ffmpeg"""
    return _ffmpeg_encode(pcm_float, "opus")


def pcm_to_aac(pcm_float: np.ndarray) -> bytes:
    """float32 PCM → AAC (ADTS) via ffmpeg"""
    return _ffmpeg_encode(pcm_float, "aac")


def pcm_to_format(pcm_float: np.ndarray, fmt: str) -> bytes:
    """根据格式名选择编码器"""
    fmt_map = {
        "wav": pcm_to_wav,
        "pcm": pcm_to_raw,
        "mp3": pcm_to_mp3,
        "opus": pcm_to_opus,
        "flac": pcm_to_flac,
        "aac": pcm_to_aac,
    }
    fn = fmt_map.get(fmt.lower(), pcm_to_wav)
    return fn(pcm_float)


def _ffmpeg_encode(pcm_float: np.ndarray, fmt: str) -> bytes:
    """使用 ffmpeg 子进程编码"""
    pcm_int16 = (pcm_float * 32767).clip(-32768, 32767).astype(np.int16)

    codec_map = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac"}
    format_map = {"mp3": "mp3", "opus": "ogg", "aac": "adts"}

    codec = codec_map.get(fmt, "copy")
    container = format_map.get(fmt, fmt)

    cmd = [
        "ffmpeg", "-y",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", "pipe:0",
        "-c:a", codec,
        "-f", container,
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd, input=pcm_int16.tobytes(),
        capture_output=True, timeout=30,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 编码失败: {proc.stderr.decode()}")
    return proc.stdout


def wav_header_placeholder() -> bytes:
    """生成 44 字节 WAV 头（大小占位，用于流式传输）"""
    header = bytearray(44)
    # RIFF
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, 0xFFFFFFFF)  # file size - 8 (占位)
    header[8:12] = b"WAVE"
    # fmt
    header[12:16] = b"fmt "
    struct.pack_into("<I", header, 16, 16)  # chunk size
    struct.pack_into("<H", header, 20, 1)   # PCM
    struct.pack_into("<H", header, 22, 1)   # mono
    struct.pack_into("<I", header, 24, SAMPLE_RATE)  # sample rate
    struct.pack_into("<I", header, 28, SAMPLE_RATE * 2)  # byte rate
    struct.pack_into("<H", header, 32, 2)   # block align
    struct.pack_into("<H", header, 34, 16)  # bits per sample
    # data
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, 0xFFFFFFFF)  # data size (占位)
    return bytes(header)


def media_type_for_format(fmt: str) -> str:
    """返回格式对应的 MIME 类型"""
    mt_map = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "flac": "audio/flac",
        "aac": "audio/aac",
    }
    return mt_map.get(fmt.lower(), "audio/wav")
