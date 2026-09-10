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

def apply_vad(audio, sr=16000, mode=3):
    vad = webrtcvad.Vad(mode)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    frame_ms = 30
    frame_len = int(sr * (frame_ms / 1000.0)) * 2 # 2 bytes per sample
    
    out_audio = audio.copy()
    num_frames = len(pcm16) // frame_len
    
    for i in range(num_frames):
        frame_bytes = pcm16[i * frame_len : (i + 1) * frame_len]
        is_speech = vad.is_speech(frame_bytes, sr)
        if not is_speech:
            start_sample = i * (sr * frame_ms // 1000)
            end_sample = (i + 1) * (sr * frame_ms // 1000)
            out_audio[start_sample:end_sample] = 0.0
            
    return out_audio.astype(np.float32)

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
        return audio
    elif condition == "opus_16k":
        return apply_opus(audio, sr, bitrate="16k", dtx=False)
    elif condition == "opus_8k":
        return apply_opus(audio, sr, bitrate="8k", dtx=False)
    elif condition == "opus_16k_dtx":
        # DTX simulation: opus voip application + gating extreme low energy
        audio_opus = apply_opus(audio, sr, bitrate="16k", dtx=True)
        return apply_vad(audio_opus, sr, mode=2)
    elif condition == "vad_agg3":
        return apply_vad(audio, sr, mode=3)
    elif condition == "denoise":
        return apply_denoise(audio, sr)
    elif condition == "agc":
        return apply_agc(audio, sr, target_lufs=-23.0)
    elif condition == "full_chain":
        # denoise -> AGC -> opus_16k + DTX
        a_den = apply_denoise(audio, sr)
        a_agc = apply_agc(a_den, sr, target_lufs=-23.0)
        a_opus = apply_opus(a_agc, sr, bitrate="16k", dtx=True)
        return apply_vad(a_opus, sr, mode=3)
    else:
        raise ValueError(f"Unknown condition: {condition}")

def compute_silence_removal_statistic(clean_audio, degraded_audio, sr=16000, frame_ms=20):
    frame_len = int(sr * (frame_ms / 1000.0))
    min_len = min(len(clean_audio), len(degraded_audio))
    clean_audio = clean_audio[:min_len]
    degraded_audio = degraded_audio[:min_len]
    
    n_frames = min_len // frame_len
    if n_frames == 0:
        return 0.0
        
    clean_rms = []
    deg_rms = []
    
    for i in range(n_frames):
        c_f = clean_audio[i * frame_len : (i + 1) * frame_len]
        d_f = degraded_audio[i * frame_len : (i + 1) * frame_len]
        clean_rms.append(np.sqrt(np.mean(c_f ** 2) + 1e-12))
        deg_rms.append(np.sqrt(np.mean(d_f ** 2) + 1e-12))
        
    clean_rms = np.array(clean_rms)
    deg_rms = np.array(deg_rms)
    
    # Define silence in clean signal as bottom 30% of frame energies or RMS < 0.01
    thresh = min(0.01, np.percentile(clean_rms, 30))
    silence_mask = clean_rms <= thresh
    
    if silence_mask.sum() == 0:
        return 0.0
        
    # Silence removed/shortened: energy suppressed further or zeroed out
    silence_deg_rms = deg_rms[silence_mask]
    silence_clean_rms = clean_rms[silence_mask]
    
    # Fraction of silent frames where degraded energy is < 50% of clean energy or < 1e-4
    suppressed = (silence_deg_rms < 0.5 * silence_clean_rms) | (silence_deg_rms < 1e-4)
    return float(suppressed.sum() / silence_mask.sum())

def generate_and_cache_degraded_audio(clip_row, condition, degraded_dir):
    clip_uid = clip_row["clip_uid"]
    out_wav = os.path.join(degraded_dir, condition, f"{clip_uid}.wav")
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    
    if os.path.exists(out_wav):
        return out_wav
        
    clean_audio, sr = load_audio_16k(clip_row["file_path"])
    deg_audio = process_degradation(clean_audio, condition, sr=sr)
    sf.write(out_wav, deg_audio, sr)
    return out_wav

if __name__ == "__main__":
    import yaml
    with open("icassp/config.yaml") as f:
        cfg = yaml.safe_load(f)
    print("Testing degradation module...")
    clean, sr = load_audio_16k("/Users/anishapattanayak/Documents/SLT/Dysfluency-Aware-Endpointing/clips/HeStutters/20/249.wav")
    print("Clean audio shape:", clean.shape)
    for cond in cfg["degradation_conditions"]:
        deg = process_degradation(clean, cond, sr=sr)
        stat = compute_silence_removal_statistic(clean, deg, sr=sr)
        print(f"Condition: {cond:15s} | Deg audio shape: {deg.shape} | Silence removal stat: {stat:.4f}")
