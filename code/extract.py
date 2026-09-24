import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WavLMModel, AutoFeatureExtractor
from degrade import load_audio_16k, process_degradation, ensure_min_duration
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

def _process_one_clip_helper(args):
    uid, file_path, condition, degraded_dir = args
    if condition == "clean":
        raw_audio, sr = load_audio_16k(file_path)
    else:
        cached_wav = os.path.join(degraded_dir, condition, f"{uid}.wav")
        if os.path.exists(cached_wav):
            raw_audio, sr = load_audio_16k(cached_wav)
        else:
            clean_audio, sr = load_audio_16k(file_path)
            raw_audio, _ = process_degradation(clean_audio, condition, sr=sr)
            os.makedirs(os.path.dirname(cached_wav), exist_ok=True)
            import soundfile as sf
            sf.write(cached_wav, raw_audio, sr)
            
    true_len = len(raw_audio)
    padded_audio, was_padded = ensure_min_duration(raw_audio, sr=16000, min_ms=400)
    return uid, padded_audio, true_len, was_padded

def extract_features_for_subset(df, condition, corpus="sep28k", config_path="icassp/config.yaml", force_reextract=False, layer_indices=None):
    cfg = load_config(config_path)
    cache_dir = cfg["paths"]["cache_dir"]
    degraded_dir = cfg["paths"]["degraded_audio_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_ver = cfg.get("cache_version", "v4")
    if layer_indices is None:
        target_layers = list(range(13))
    else:
        target_layers = list(layer_indices)
        
    target_uids = df["clip_uid"].tolist()
    index_file = os.path.join(cache_dir, f"{corpus}_{condition}_{cache_ver}_clip_ids.json")
    
    cached_layer_files = {
        l: os.path.join(cache_dir, f"{corpus}_{condition}_{cache_ver}_layer{l}.npy")
        for l in target_layers
    }
    
    if not force_reextract and os.path.exists(index_file) and all(os.path.exists(cached_layer_files[l]) for l in target_layers):
        print(f"[extract] Cache hit for corpus={corpus}, condition={condition} ({cache_ver}). Checking alignment...")
        with open(index_file) as f:
            cached_uids = json.load(f)
            
        if cached_uids == target_uids:
            layer_dict = {l: np.load(cached_layer_files[l]) for l in target_layers}
            return layer_dict, cached_uids
        elif set(cached_uids) == set(target_uids):
            print(f"[extract] Reordering cached features to match dataframe index...")
            uid_to_idx = {uid: i for i, uid in enumerate(cached_uids)}
            reorder_idx = [uid_to_idx[uid] for uid in target_uids]
            layer_dict = {}
            for l in target_layers:
                raw_mat = np.load(cached_layer_files[l])
                layer_dict[l] = raw_mat[reorder_idx]
            return layer_dict, target_uids
        else:
            print(f"[extract] Cache UID set mismatch. Re-extracting...")

    print(f"[extract] Extracting features for corpus={corpus}, condition={condition} ({len(df)} clips, layers={target_layers})...")
    model, extractor, device = get_wavlm_model(cfg["model"]["encoder"])
    
    batch_size = cfg["model"].get("batch_size", 32)
    
    import concurrent.futures
    tasks = [(r["clip_uid"], r["file_path"], condition, degraded_dir) for _, r in df.iterrows()]
    max_workers = min(8, os.cpu_count() or 4)
    
    print(f"[extract] Loading & degrading audio in parallel with {max_workers} processes...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_process_one_clip_helper, tasks), total=len(tasks), desc=f"Audio prep ({condition})"))
        
    uid_to_res = {r[0]: (r[1], r[2], r[3]) for r in results}
    audios_raw = [uid_to_res[uid][0] for uid in target_uids]
    true_lengths = [uid_to_res[uid][1] for uid in target_uids]
    was_padded_flags = [uid_to_res[uid][2] for uid in target_uids]
        
    # Sort clips by length before batching to optimize batch padding
    clip_records = [
        (idx, uid, a, tlen, wpad)
        for idx, (uid, a, tlen, wpad) in enumerate(zip(target_uids, audios_raw, true_lengths, was_padded_flags))
    ]
    sorted_records = sorted(clip_records, key=lambda x: len(x[2]))
    
    sorted_uids = [r[1] for r in sorted_records]
    orig_indices = [r[0] for r in sorted_records]
    inv_order = np.argsort(orig_indices)
    
    layer_feats_batch = {l: [] for l in target_layers}
    
    for i in tqdm(range(0, len(sorted_records), batch_size), desc=f"Inference ({condition})"):
        batch_records = sorted_records[i : i + batch_size]
        batch_audios = [r[2] for r in batch_records]
        batch_true_lens = np.array([r[3] for r in batch_records])
        lengths = np.array([len(a) for a in batch_audios])
        max_len = int(lengths.max())
        
        padded = np.zeros((len(batch_audios), max_len), dtype=np.float32)
        attn = np.zeros((len(batch_audios), max_len), dtype=np.int64)
        
        for b, a in enumerate(batch_audios):
            padded[b, :len(a)] = a
            eff_len = max(min(len(a), 6400), batch_true_lens[b])
            attn[b, :min(eff_len, len(a))] = 1
            
        inputs = torch.tensor(padded, device=device)
        attn_t = torch.tensor(attn, device=device)
        
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attn_t, output_hidden_states=True)
            feat_lens = model._get_feat_extract_output_lengths(
                torch.tensor(np.maximum(1, batch_true_lens))
            ).numpy().astype(int)
            
            for l_idx in target_layers:
                hs = outputs.hidden_states[l_idx]
                T = hs.shape[1]
                fmask = torch.zeros(hs.shape[0], T, device=hs.device)
                for b, fl in enumerate(feat_lens):
                    valid_frames = min(max(int(fl), 1), T)
                    fmask[b, :valid_frames] = 1.0
                    
                pooled = (hs * fmask.unsqueeze(-1)).sum(dim=1) / fmask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                layer_feats_batch[l_idx].append(pooled.cpu().numpy())
                
        del inputs, attn_t, outputs
        if device.type == "mps":
            torch.mps.empty_cache()
                
    # Concatenate per layer and restore original order
    final_layer_dict = {}
    for l in target_layers:
        concat_mat = np.concatenate(layer_feats_batch[l], axis=0)
        restored_mat = concat_mat[inv_order]
        final_layer_dict[l] = restored_mat
        np.save(cached_layer_files[l], restored_mat)
        
    with open(index_file, "w") as f:
        json.dump(target_uids, f)
        
    # Save padding metadata for sensitivity analysis
    pad_meta_file = os.path.join(cache_dir, f"{corpus}_{condition}_{cache_ver}_padding_meta.json")
    with open(pad_meta_file, "w") as f:
        json.dump({uid: bool(flag) for uid, flag in zip(target_uids, was_padded_flags)}, f)
        
    print(f"[extract] Successfully cached features for condition={condition} to {cache_dir}")
    return final_layer_dict, target_uids
