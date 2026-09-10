import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WavLMModel, AutoFeatureExtractor
from degrade import load_audio_16k, process_degradation, generate_and_cache_degraded_audio
from prep import load_config

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

_WAVLM_MODEL = None
_FEATURE_EXTRACTOR = None

def get_wavlm_model(model_name="microsoft/wavlm-base-plus"):
    global _WAVLM_MODEL, _FEATURE_EXTRACTOR
    device = get_device()
    if _WAVLM_MODEL is None:
        print(f"[extract] Loading {model_name} on {device}...")
        _FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(model_name)
        _WAVLM_MODEL = WavLMModel.from_pretrained(model_name).to(device)
        _WAVLM_MODEL.eval()
        for param in _WAVLM_MODEL.parameters():
            param.requires_grad = False
    return _WAVLM_MODEL, _FEATURE_EXTRACTOR, device

def extract_features_for_subset(df, condition, corpus="sep28k", config_path="icassp/config.yaml", force_reextract=False):
    cfg = load_config(config_path)
    cache_dir = cfg["paths"]["cache_dir"]
    degraded_dir = cfg["paths"]["degraded_audio_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_ver = cfg.get("cache_version", "v2")
    n_layers = 13
    clip_uids = df["clip_uid"].tolist()
    index_file = os.path.join(cache_dir, f"{corpus}_{condition}_{cache_ver}_clip_ids.json")
    
    cached_layer_files = [
        os.path.join(cache_dir, f"{corpus}_{condition}_{cache_ver}_layer{l}.npy")
        for l in range(n_layers)
    ]
    
    if not force_reextract and os.path.exists(index_file) and all(os.path.exists(f) for f in cached_layer_files):
        print(f"[extract] Cache hit for corpus={corpus}, condition={condition} ({cache_ver}). Loading from disk...")
        layer_dict = {}
        for l in range(n_layers):
            layer_dict[l] = np.load(cached_layer_files[l])
        with open(index_file) as f:
            cached_uids = json.load(f)
        return layer_dict, cached_uids

    print(f"[extract] Extracting features for corpus={corpus}, condition={condition} ({len(df)} clips)...")
    model, extractor, device = get_wavlm_model(cfg["model"]["encoder"])
    
    layer_feats = {l: [] for l in range(n_layers)}
    batch_size = cfg["model"].get("batch_size", 16)
    
    audios = []
    processed_uids = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading audio ({condition})"):
        uid = row["clip_uid"]
        if condition == "clean":
            wav_path = row["file_path"]
            audio, sr = load_audio_16k(wav_path)
        else:
            cached_wav = os.path.join(degraded_dir, condition, f"{uid}.wav")
            if os.path.exists(cached_wav):
                audio, sr = load_audio_16k(cached_wav)
            else:
                raw_audio, sr = load_audio_16k(row["file_path"])
                audio = process_degradation(raw_audio, condition, sr=sr)
                os.makedirs(os.path.dirname(cached_wav), exist_ok=True)
                sf_write_audio(cached_wav, audio, sr)
                
        audios.append(audio)
        processed_uids.append(uid)
        
    for i in range(0, len(audios), batch_size):
        batch_audios = audios[i : i + batch_size]
        target_samples = 48000
        padded = np.zeros((len(batch_audios), target_samples), dtype=np.float32)
        for b_idx, a in enumerate(batch_audios):
            if len(a) >= target_samples:
                padded[b_idx] = a[:target_samples]
            else:
                padded[b_idx, :len(a)] = a
                
        inputs = torch.tensor(padded).to(device)
        
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            for l_idx, hs in enumerate(hidden_states):
                pooled = hs.mean(dim=1).cpu().numpy()
                layer_feats[l_idx].append(pooled)
                
    final_layer_dict = {}
    for l in range(n_layers):
        final_layer_dict[l] = np.concatenate(layer_feats[l], axis=0)
        np.save(cached_layer_files[l], final_layer_dict[l])
        
    with open(index_file, "w") as f:
        json.dump(processed_uids, f)
        
    print(f"[extract] Successfully cached features for condition={condition} to {cache_dir}")
    return final_layer_dict, processed_uids

def sf_write_audio(path, audio, sr):
    import soundfile as sf
    sf.write(path, audio, sr)
