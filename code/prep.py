import os
import yaml
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(os.path.dirname(script_dir), os.path.basename(config_path)),
            os.path.join(script_dir, os.path.basename(config_path)),
            os.path.join(os.getcwd(), config_path),
            os.path.join(os.getcwd(), "icassp", config_path),
            os.path.join(os.getcwd(), os.path.basename(config_path)),
            os.path.join(os.getcwd(), "icassp", os.path.basename(config_path)),
        ]
        for c in candidates:
            if os.path.exists(c):
                config_path = c
                break
            
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    config_dir = os.path.dirname(os.path.abspath(config_path))
    local_cfg_candidates = [
        os.path.join(config_dir, "config.local.yaml"),
        os.path.join(os.path.dirname(config_dir), "config.local.yaml"),
        os.path.join(os.getcwd(), "config.local.yaml"),
        os.path.join(os.getcwd(), "icassp", "config.local.yaml")
    ]
    for local_cfg_path in local_cfg_candidates:
        if os.path.exists(local_cfg_path):
            with open(local_cfg_path, "r") as f:
                local_cfg = yaml.safe_load(f)
                if local_cfg:
                    if "paths" in local_cfg and "paths" in cfg:
                        cfg["paths"].update(local_cfg["paths"])
                    for k, v in local_cfg.items():
                        if k != "paths":
                            cfg[k] = v
            break
                        
    return cfg

def load_and_filter_sep28k(config):
    labels_path = config["paths"]["raw_labels_path"]
    clips_dir = config["paths"]["raw_clips_dir"]
    
    df = pd.read_csv(labels_path)
    
    # Map file path
    df["file_path"] = df.apply(
        lambda r: os.path.join(clips_dir, str(r["Show"]), str(r["EpId"]), f"{r['ClipId']}.wav"),
        axis=1
    )
    df["exists"] = df["file_path"].apply(os.path.exists)
    df = df[df["exists"]].copy()
    
    # Drop NoSpeech and Music
    df = df[(df["NoSpeech"] == 0) & (df["Music"] == 0)].copy()
    
    # Ensure raw counts are ints
    count_cols = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection", "NoStutteredWords"]
    for c in count_cols:
        df[c] = df[c].astype(int)
        
    df["episode_id"] = df["Show"].astype(str) + "_" + df["EpId"].astype(str)
    df["clip_uid"] = df["Show"].astype(str) + "_" + df["EpId"].astype(str) + "_" + df["ClipId"].astype(str)
    
    return df

def build_working_subset(df, config, subset_size=8000, seed=42):
    """
    Constructs a deterministic, class-balanced subset of clips from SEP-28k.
    If subset_size == 8000:
        Extracts 1,300 clips per dysfluency class (rare classes first) and 1,500 fluent clips
        to produce an exact N = 8,000 class-balanced evaluation subset.
    If subset_size is 'full' or >= len(df):
        Retains all available filtered clips.
    """
    if subset_size is None or str(subset_size).lower() == "full":
        return df.copy().reset_index(drop=True)
        
    target_n = int(subset_size)
    if target_n == 8000:
        classes = ['WordRep', 'SoundRep', 'Prolongation', 'Block', 'Interjection']
        rng = np.random.RandomState(seed)
        
        sampled_indices = []
        for c in classes:
            pos_idx = df[(df[c] >= 1) & (~df.index.isin(sampled_indices))].index.tolist()
            n_take = min(len(pos_idx), 1300)
            chosen = rng.choice(pos_idx, size=n_take, replace=False)
            sampled_indices.extend(chosen)
            
        is_stutter = (df[['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']] >= 1).any(axis=1)
        fluent_idx = df[(~is_stutter) & (~df.index.isin(sampled_indices))].index.tolist()
        needed = target_n - len(sampled_indices)
        chosen_fluent = rng.choice(fluent_idx, size=needed, replace=False)
        sampled_indices.extend(chosen_fluent)
        
        subset_df = df.loc[sampled_indices].copy().reset_index(drop=True)
        return subset_df
    else:
        # Fallback proportional sampling
        hard_thresh = config.get("hard_thresh", 1)
        is_stutter = (df[['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']] >= hard_thresh).any(axis=1)
        stutter_df = df[is_stutter].copy()
        fluent_df = df[~is_stutter].copy()
        
        n_stutter = min(len(stutter_df), int(target_n * 0.8))
        n_fluent = target_n - n_stutter
        
        sample_s = stutter_df.sample(n=n_stutter, random_state=seed)
        sample_f = fluent_df.sample(n=n_fluent, random_state=seed)
        return pd.concat([sample_s, sample_f]).reset_index(drop=True)

def add_group_kfold_splits(df, n_folds=5, seed=42):
    gkf = GroupKFold(n_splits=n_folds)
    df["fold"] = -1
    groups = df["episode_id"].values
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        df.loc[val_idx, "fold"] = fold
        
    return df

def add_cross_show_splits(df):
    """Cross-show split: hold out HVSA and MyStutteringLife."""
    held_out_shows = ["HVSA", "MyStutteringLife"]
    df["cross_show_split"] = df["Show"].apply(lambda s: "test" if s in held_out_shows else "train")
    return df

def prepare_dataset(config_path="icassp/config.yaml", subset_csv=None, seed=42, force_rebuild=False):
    config = load_config(config_path)
    cache_dir = config["paths"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    
    manifest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_csv = os.path.join(manifest_dir, "dataset_manifest.csv")
    
    if subset_csv is None:
        subset_csv = os.path.join(cache_dir, "sep28k_subset.csv")
        
    if not force_rebuild and os.path.exists(subset_csv):
        df_existing = pd.read_csv(subset_csv)
        if len(df_existing) == 8000 and "fold" in df_existing.columns:
            print(f"[prep] Loading existing valid 8,000-clip subset from {subset_csv}")
            return df_existing
            
    print("[prep] Filtering SEP-28k and creating class-balanced N=8,000 working subset...")
    raw_df = load_and_filter_sep28k(config)
    subset_df = build_working_subset(raw_df, config, subset_size=8000, seed=seed)
    subset_df = add_group_kfold_splits(subset_df, n_folds=config.get("n_folds", 5), seed=seed)
    subset_df = add_cross_show_splits(subset_df)
    
    # Add explicit majority-vote and any-annotator indicator columns
    target_cols = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
    for col in target_cols:
        subset_df[f"{col}_any"] = (subset_df[col] >= 1).astype(int)
        subset_df[f"{col}_maj"] = (subset_df[col] >= 2).astype(int)
        
    subset_df.to_csv(subset_csv, index=False)
    subset_df.to_csv(manifest_csv, index=False)
    print(f"[prep] Saved class-balanced N={len(subset_df)} subset to {subset_csv} and {manifest_csv}")
    return subset_df

if __name__ == "__main__":
    df = prepare_dataset()
    print("Subset shape:", df.shape)
    print("Fold distribution:\n", df["fold"].value_counts())
    print("Cross-show distribution:\n", df["cross_show_split"].value_counts())
