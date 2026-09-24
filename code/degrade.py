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

def apply_opus(audio, sr=16000, bitrate="16k", dtx=False, application=None):
    """
    Encode and decode audio with libopus via ffmpeg.
    Note: ffmpeg libopus uses -application voip / audio.
    The parameter dtx=True historically mapped to -application voip.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_wav = os.path.join(tmpdir, "in.wav")
        out_ogg = os.path.join(tmpdir, "out.ogg")
        out_wav = os.path.join(tmpdir, "out.wav")
        
        sf.write(in_wav, audio, sr)
        
        cmd = ["ffmpeg", "-y", "-i", in_wav, "-c:a", "libopus", "-b:a", bitrate]
        if application is not None:
            cmd.extend(["-application", application])
        elif dtx:
            # VoIP profile (historically designated dtx in prior drafts)
            cmd.extend(["-application", "voip"])
        else:
            cmd.extend(["-application", "audio"])
        cmd.append(out_ogg)
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        cmd2 = ["ffmpeg", "-y", "-i", out_ogg, "-ar", "16000", "-ac", "1", out_wav]
        subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        deg_audio, _ = sf.read(out_wav)
        return deg_audio.astype(np.float32)

def apply_vad(audio, sr=16000, mode=3, remove=True, hangover_frames=0, return_meta=False):
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
            
    min_samples = int(sr * 0.4)
    if not keep:
        if not remove:
            # Zeroing preserves full waveform length (duration integrity)
            out_audio = np.zeros_like(audio)
            actual_retained_samples = 0
            completely_rejected = True
        else:
            # Completely rejected clip when removing frames:
            # Return standardized 400ms silence so encoder receptive field is satisfied,
            # but record actual_retained_samples = 0 and completely_rejected = True.
            out_audio = np.zeros(min_samples, dtype=np.float32)
            actual_retained_samples = 0
            completely_rejected = True
    else:
        if remove:
            out_audio = np.concatenate(keep).astype(np.float32)
            actual_retained_samples = len(out_audio)
            completely_rejected = False
        else:
            out_audio = audio.copy()
            speech_samples = 0
            for i, f in enumerate(flags):
                is_sp = f or (hangover_frames > 0 and any(flags[max(0, i - hangover_frames) : i]))
                if not is_sp:
                    out_audio[i * n : (i + 1) * n] = 0.0
                else:
                    speech_samples += n
            actual_retained_samples = speech_samples
            completely_rejected = False

    meta = {
        "actual_retained_samples": int(actual_retained_samples),
        "completely_rejected": bool(completely_rejected),
        "was_padded": len(out_audio) < min_samples or (actual_retained_samples == 0 and remove),
        "duration_reduction_frac": float(np.clip(1.0 - actual_retained_samples / max(len(audio), 1), 0.0, 1.0)),
        "retained_duration_ratio": float(np.clip(actual_retained_samples / max(len(audio), 1), 0.0, 1.0))
    }
    
    if return_meta:
        return out_audio, meta
    return out_audio

def apply_vad_with_meta(audio, sr=16000, mode=3, remove=True, hangover_frames=0):
    """Convenience helper returning (audio, metadata)."""
    return apply_vad(audio, sr=sr, mode=mode, remove=remove, hangover_frames=hangover_frames, return_meta=True)

def apply_random_deletion(audio, target_removal_frac=0.3, frame_ms=30, sr=16000, seed=42, return_meta=False):
    """
    Exploratory fixed-fraction control: randomly deletes chunks totaling target_removal_frac.
    """
    n = int(sr * frame_ms / 1000)
    num_frames = len(audio) // n
    min_samples = int(sr * 0.4)
    if num_frames == 0:
        meta = {
            "actual_retained_samples": len(audio),
            "completely_rejected": False,
            "was_padded": len(audio) < min_samples,
            "duration_reduction_frac": 0.0,
            "retained_duration_ratio": 1.0
        }
        return (audio.copy(), meta) if return_meta else audio.copy()
    
    rng = np.random.RandomState(seed)
    num_to_remove = int(round(num_frames * target_removal_frac))
    num_to_remove = max(0, min(num_frames, num_to_remove))
    
    drop_indices = set(rng.choice(num_frames, size=num_to_remove, replace=False))
    keep = [audio[i * n : (i + 1) * n] for i in range(num_frames) if i not in drop_indices]
    
    if not keep:
        out_audio = np.zeros(min_samples, dtype=np.float32)
        actual_retained_samples = 0
        completely_rejected = True
    else:
        rem = audio[num_frames * n :]
        kept_audio = np.concatenate(keep)
        if len(rem) > 0:
            kept_audio = np.concatenate([kept_audio, rem])
        out_audio = kept_audio.astype(np.float32)
        actual_retained_samples = len(out_audio)
        completely_rejected = False

    meta = {
        "actual_retained_samples": int(actual_retained_samples),
        "completely_rejected": bool(completely_rejected),
        "was_padded": len(out_audio) < min_samples or completely_rejected,
        "duration_reduction_frac": float(np.clip(1.0 - actual_retained_samples / max(len(audio), 1), 0.0, 1.0)),
        "retained_duration_ratio": float(np.clip(actual_retained_samples / max(len(audio), 1), 0.0, 1.0))
    }
    if return_meta:
        return out_audio, meta
    return out_audio

def apply_random_deletion_matched(audio, vad_silence_frac, frame_ms=30, sr=16000, seed=42, return_meta=False):
    """
    Per-clip duration-matched random deletion control:
    Deletes the exact fraction of 30ms frames that VAD removed from this specific clip,
    distributed uniformly at random using a deterministic seed.
    """
    return apply_random_deletion(
        audio, target_removal_frac=vad_silence_frac, frame_ms=frame_ms, sr=sr, seed=seed, return_meta=return_meta
    )

def apply_endpoint_truncate(audio, sr=16000, silence_thresh_ms=800, mode=1, frame_ms=30):
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

def process_degradation(audio, condition, sr=16000, clip_uid=None, vad_silence_frac=None):
    import hashlib
    seed = int(hashlib.md5((clip_uid or "default").encode()).hexdigest(), 16) % 100000 + 42 if clip_uid else 42
    
    meta = {
        "actual_retained_samples": len(audio),
        "completely_rejected": False,
        "was_padded": False,
        "duration_reduction_frac": 0.0,
        "retained_duration_ratio": 1.0
    }
    
    if condition == "clean":
        deg = audio.astype(np.float32)
    elif condition == "opus_16k":
        deg = apply_opus(audio, sr, bitrate="16k", dtx=False, application="audio")
    elif condition == "opus_8k":
        deg = apply_opus(audio, sr, bitrate="8k", dtx=False, application="audio")
    elif condition in ("opus_16k_voip", "opus_16k_dtx"):
        # VoIP application mode in libopus
        deg = apply_opus(audio, sr, bitrate="16k", application="voip")
    elif condition == "vad_agg3":
        deg, meta = apply_vad(audio, sr, mode=3, remove=True, return_meta=True)
    elif condition == "vad_mode1":
        deg, meta = apply_vad(audio, sr, mode=1, remove=True, return_meta=True)
    elif condition == "vad_mode2":
        deg, meta = apply_vad(audio, sr, mode=2, remove=True, return_meta=True)
    elif condition == "vad_hangover2":
        deg, meta = apply_vad(audio, sr, mode=3, remove=True, hangover_frames=2, return_meta=True)
    elif condition == "vad_zero":
        deg, meta = apply_vad(audio, sr, mode=3, remove=False, return_meta=True)
    elif condition == "endpoint_800ms":
        deg = apply_endpoint_truncate(audio, sr, silence_thresh_ms=800, mode=1)
        meta["actual_retained_samples"] = len(deg)
    elif condition == "endpoint_1200ms":
        deg = apply_endpoint_truncate(audio, sr, silence_thresh_ms=1200, mode=1)
        meta["actual_retained_samples"] = len(deg)
    elif condition == "denoise":
        deg = apply_denoise(audio, sr)
    elif condition == "agc":
        deg = apply_agc(audio, sr, target_lufs=-23.0)
    elif condition == "full_chain_novad":
        # Control condition: full pipeline excluding VAD stage
        a_den = apply_denoise(audio, sr)
        a_agc = apply_agc(a_den, sr, target_lufs=-23.0)
        deg = apply_opus(a_agc, sr, bitrate="16k", application="voip")
    elif condition == "full_chain":
        a_den = apply_denoise(audio, sr)
        a_agc = apply_agc(a_den, sr, target_lufs=-23.0)
        a_opus = apply_opus(a_agc, sr, bitrate="16k", application="voip")
        deg, meta = apply_vad(a_opus, sr, mode=3, remove=True, return_meta=True)
    elif condition == "random_del_matched":
        if vad_silence_frac is None:
            _, v_meta = apply_vad(audio, sr, mode=3, remove=True, return_meta=True)
            vad_silence_frac = v_meta["duration_reduction_frac"]
        deg, meta = apply_random_deletion_matched(audio, vad_silence_frac=vad_silence_frac, sr=sr, seed=seed, return_meta=True)
    elif condition == "random_del_30pct":
        deg, meta = apply_random_deletion(audio, target_removal_frac=0.30, sr=sr, seed=seed, return_meta=True)
    elif condition.startswith("random_del_"):
        pct_str = condition.replace("random_del_", "").replace("pct", "")
        frac = float(pct_str) / 100.0
        deg, meta = apply_random_deletion(audio, target_removal_frac=frac, sr=sr, seed=seed, return_meta=True)
    else:
        raise ValueError(f"Unknown condition: {condition}")
        
    deg_unpadded = deg.astype(np.float32)
    min_samples = int(sr * 0.4)
    was_padded = len(deg_unpadded) < min_samples or meta.get("completely_rejected", False)
    meta["was_padded"] = was_padded
    meta["actual_retained_samples"] = meta.get("actual_retained_samples", len(deg_unpadded))
    return deg_unpadded, was_padded

def _frame_rms(x, sr, frame_ms):
    n = int(sr * frame_ms / 1000)
    if len(x) < n:
        return np.array([])
    return np.array([np.sqrt(np.mean(x[i * n : (i + 1) * n] ** 2) + 1e-12) for i in range(len(x) // n)])

def compute_silence_removal_statistic(clean, degraded, sr=16000, frame_ms=20, actual_retained_samples=None):
    rms_c = _frame_rms(clean, sr, frame_ms)
    rms_d = _frame_rms(degraded, sr, frame_ms)
    if rms_c.size == 0:
        return {"silence_removed_frac": 0.0, "duration_reduction_frac": 0.0, "silence_thresh": 0.0}
    thr = max(0.01, float(np.percentile(rms_c, 30)))
    sil_c = (rms_c <= thr).sum() * frame_ms / 1000.0
    sil_d = (rms_d <= thr).sum() * frame_ms / 1000.0
    
    if actual_retained_samples is not None:
        dur_loss = float(np.clip(1.0 - (actual_retained_samples / sr) / max(len(clean) / sr, 1e-6), 0.0, 1.0))
    else:
        dur_loss = float(np.clip(1.0 - (len(degraded) / sr) / max(len(clean) / sr, 1e-6), 0.0, 1.0))
        
    return {
        "silence_removed_frac": float(np.clip(1.0 - sil_d / max(sil_c, 1e-6), 0.0, 1.0)),
        "duration_reduction_frac": dur_loss,
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
