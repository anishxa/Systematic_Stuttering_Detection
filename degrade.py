import os
import glob
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import webrtcvad
import noisereduce as nr
import pandas as pd

def load_audio_16k(wav_path):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio.astype(np.float32), sr

def apply_opus(audio, sr=16000, bitrate="16k", dtx=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        in_wav = os.path.join(tmpdir, "in.wav")
        out_ogg = os.path.join(tmpdir, "out.ogg")
        out_wav = os.path.join(tmpdir, "out.wav")
        
        sf.write(in_wav, audio, sr)
        
        cmd = ["ffmpeg", "-y", "-i", in_wav, "-c:a", "libopus", "-b:a", bitrate]
        if dtx:
            cmd.extend(["-application", "voip"])
        else:
            cmd.extend(["-application", "audio"])
        cmd.append(out_ogg)
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        cmd2 = ["ffmpeg", "-y", "-i", out_ogg, "-ar", "16000", "-ac", "1", out_wav]
        subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        deg_audio, _ = sf.read(out_wav)
        return deg_audio.astype(np.float32)

def apply_vad(audio, sr=16000, mode=3, remove=True, hangover_frames=0):
    vad = webrtcvad.Vad(mode)
    frame_ms = 30
    n = int(sr * frame_ms / 1000)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    
    keep = []
    flags = []
    num_frames = len(pcm16) // n
    
    for i in range(num_frames):
        frame_bytes = pcm16[i * n : (i + 1) * n].tobytes()
        flags.append(vad.is_speech(frame_bytes, sr))
        
    for i, f in enumerate(flags):
        is_sp = f or (hangover_frames > 0 and any(flags[max(0, i - hangover_frames) : i]))
        if is_sp:
            keep.append(audio[i * n : (i + 1) * n])
            
    if not keep:
        return np.zeros(n, dtype=np.float32)
        
    if remove:
        return np.concatenate(keep).astype(np.float32)
    else:
        out_audio = audio.copy()
        for i, f in enumerate(flags):
            is_sp = f or (hangover_frames > 0 and any(flags[max(0, i - hangover_frames) : i]))
            if not is_sp:
                out_audio[i * n : (i + 1) * n] = 0.0
        return out_audio.astype(np.float32)

def apply_endpoint_truncate(audio, sr=16000, silence_thresh_ms=300, mode=2, frame_ms=30):
    vad = webrtcvad.Vad(mode)
    n = int(sr * frame_ms / 1000)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    need = max(1, int(np.ceil(silence_thresh_ms / frame_ms)))
    consec, seen_speech = 0, False
    for i in range(len(pcm) // n):
        frame_bytes = pcm[i * n : (i + 1) * n].tobytes()
        if vad.is_speech(frame_bytes, sr):
            seen_speech, consec = True, 0
        else:
            consec += 1
            if seen_speech and consec >= need:
                return audio[: (i - need + 1) * n].astype(np.float32)
    return audio.astype(np.float32)

def ensure_min_duration(audio, sr=16000, min_ms=400):
    min_samples = int(sr * min_ms / 1000)
    if len(audio) < min_samples:
        padded = np.zeros(min_samples, dtype=np.float32)
        if len(audio) > 0:
            padded[:len(audio)] = audio
        return padded, True
    return audio.astype(np.float32), False

def apply_denoise(audio, sr=16000):
    return nr.reduce_noise(y=audio, sr=sr).astype(np.float32)

def apply_agc(audio, sr=16000, target_lufs=-23.0):
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
        if np.isinf(loudness) or np.isnan(loudness):
            return audio
        normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    except Exception:
        return audio

def process_degradation(audio, condition, sr=16000):
    if condition == "clean":
        deg = audio.astype(np.float32)
    elif condition == "opus_16k":
        deg = apply_opus(audio, sr, bitrate="16k", dtx=False)
    elif condition == "opus_8k":
        deg = apply_opus(audio, sr, bitrate="8k", dtx=False)
    elif condition == "opus_16k_dtx":
        deg = apply_opus(audio, sr, bitrate="16k", dtx=True)
    elif condition == "vad_agg3":
        deg = apply_vad(audio, sr, mode=3, remove=True)
    elif condition == "vad_zero":
        deg = apply_vad(audio, sr, mode=3, remove=False)
    elif condition == "endpoint_300ms":
        deg = apply_endpoint_truncate(audio, sr, silence_thresh_ms=300, mode=2)
    elif condition == "endpoint_500ms":
        deg = apply_endpoint_truncate(audio, sr, silence_thresh_ms=500, mode=2)
    elif condition == "denoise":
        deg = apply_denoise(audio, sr)
    elif condition == "agc":
        deg = apply_agc(audio, sr, target_lufs=-23.0)
    elif condition == "full_chain":
        a_den = apply_denoise(audio, sr)
        a_agc = apply_agc(a_den, sr, target_lufs=-23.0)
        a_opus = apply_opus(a_agc, sr, bitrate="16k", dtx=True)
        deg = apply_vad(a_opus, sr, mode=3, remove=True)
    else:
        raise ValueError(f"Unknown condition: {condition}")
        
    deg_unpadded = deg.astype(np.float32)
    min_samples = int(sr * 0.4)
    was_padded = len(deg_unpadded) < min_samples
    return deg_unpadded, was_padded

def _frame_rms(x, sr, frame_ms):
    n = int(sr * frame_ms / 1000)
    if len(x) < n:
        return np.array([])
    return np.array([np.sqrt(np.mean(x[i * n : (i + 1) * n] ** 2) + 1e-12) for i in range(len(x) // n)])

def compute_silence_removal_statistic(clean, degraded, sr=16000, frame_ms=20):
    rms_c = _frame_rms(clean, sr, frame_ms)
    rms_d = _frame_rms(degraded, sr, frame_ms)
    if rms_c.size == 0:
        return {"silence_removed_frac": 0.0, "duration_reduction_frac": 0.0, "silence_thresh": 0.0}
    thr = max(0.01, float(np.percentile(rms_c, 30)))
    sil_c = (rms_c <= thr).sum() * frame_ms / 1000.0
    sil_d = (rms_d <= thr).sum() * frame_ms / 1000.0
    return {
        "silence_removed_frac": float(np.clip(1.0 - sil_d / max(sil_c, 1e-6), 0.0, 1.0)),
        "duration_reduction_frac": float(np.clip(1.0 - (len(degraded) / sr) / max(len(clean) / sr, 1e-6), 0.0, 1.0)),
        "silence_thresh": thr,
    }

def generate_and_cache_degraded_audio(clip_row, condition, degraded_dir):
    clip_uid = clip_row["clip_uid"]
    out_wav = os.path.join(degraded_dir, condition, f"{clip_uid}.wav")
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    
    if os.path.exists(out_wav):
        deg_audio, sr = load_audio_16k(out_wav)
        min_samples = int(sr * 0.4)
        was_padded = len(deg_audio) < min_samples
        return deg_audio, was_padded
        
    clean_audio, sr = load_audio_16k(clip_row["file_path"])
    deg_audio, was_padded = process_degradation(clean_audio, condition, sr=sr)
    sf.write(out_wav, deg_audio, sr)
    return deg_audio, was_padded
