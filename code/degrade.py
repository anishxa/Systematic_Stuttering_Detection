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
        if not remove:
            # Zeroing preserves full waveform length (duration integrity)
            return np.zeros_like(audio)
        else:
            # Completely rejected clip when removing frames:
            # Return standardized 400ms silence so encoder receptive field is satisfied
            min_samples = int(sr * 0.4)
            return np.zeros(min_samples, dtype=np.float32)
        
    if remove:
        return np.concatenate(keep).astype(np.float32)
    else:
        out_audio = audio.copy()
        for i, f in enumerate(flags):
            is_sp = f or (hangover_frames > 0 and any(flags[max(0, i - hangover_frames) : i]))
            if not is_sp:
                out_audio[i * n : (i + 1) * n] = 0.0
        return out_audio.astype(np.float32)

def apply_random_deletion(audio, target_removal_frac=0.3, frame_ms=30, sr=16000, seed=42):
    """
    Control condition: randomly deletes chunks of audio totaling target_removal_frac.
    Matches the overall duration reduction of VAD while distributing deletions uniformly at random,
    testing whether acoustic removal vs. structured speech-pause removal drives degradation.
    """
    n = int(sr * frame_ms / 1000)
    num_frames = len(audio) // n
    if num_frames == 0:
        return audio.copy()
    
    rng = np.random.RandomState(seed)
    num_to_remove = int(round(num_frames * target_removal_frac))
    num_to_remove = max(0, min(num_frames - 1, num_to_remove))
    
    drop_indices = set(rng.choice(num_frames, size=num_to_remove, replace=False))
    keep = [audio[i * n : (i + 1) * n] for i in range(num_frames) if i not in drop_indices]
    
    if not keep:
        min_samples = int(sr * 0.4)
        return np.zeros(min_samples, dtype=np.float32)
        
    rem = audio[num_frames * n :]
    kept_audio = np.concatenate(keep)
    if len(rem) > 0:
        kept_audio = np.concatenate([kept_audio, rem])
    return kept_audio.astype(np.float32)

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

def process_degradation(audio, condition, sr=16000):
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
        deg = apply_vad(audio, sr, mode=3, remove=True)
    elif condition == "vad_mode1":
        deg = apply_vad(audio, sr, mode=1, remove=True)
    elif condition == "vad_mode2":
        deg = apply_vad(audio, sr, mode=2, remove=True)
    elif condition == "vad_hangover2":
        deg = apply_vad(audio, sr, mode=3, remove=True, hangover_frames=2)
    elif condition == "vad_zero":
        deg = apply_vad(audio, sr, mode=3, remove=False)
    elif condition == "endpoint_800ms":
        deg = apply_endpoint_truncate(audio, sr, silence_thresh_ms=800, mode=1)
    elif condition == "endpoint_1200ms":
        deg = apply_endpoint_truncate(audio, sr, silence_thresh_ms=1200, mode=1)
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
        deg = apply_vad(a_opus, sr, mode=3, remove=True)
    elif condition == "random_del_30pct":
        deg = apply_random_deletion(audio, target_removal_frac=0.30, sr=sr, seed=42)
    elif condition.startswith("random_del_"):
        pct_str = condition.replace("random_del_", "").replace("pct", "")
        frac = float(pct_str) / 100.0
        deg = apply_random_deletion(audio, target_removal_frac=frac, sr=sr, seed=42)
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
