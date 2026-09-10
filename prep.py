import os
import yaml
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def load_config(config_path="icassp/config.yaml"):
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, os.path.basename(config_path))
        if os.path.exists(alt_path):
            config_path = alt_path
            
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    config_dir = os.path.dirname(os.path.abspath(config_path))
    local_cfg_path = os.path.join(config_dir, "config.local.yaml")
    if os.path.exists(local_cfg_path):
        with open(local_cfg_path, "r") as f:
            local_cfg = yaml.safe_load(f)
            if local_cfg:
                if "paths" in local_cfg and "paths" in cfg:
                    cfg["paths"].update(local_cfg["paths"])
                for k, v in local_cfg.items():
                    if k != "paths":
                        cfg[k] = v
                        
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
    np.random.seed(seed)
    hard_thresh = config.get("hard_thresh", 1)
    
    # Subsets by positivity (count >= hard_thresh)
    is_block = df["Block"] >= hard_thresh
    is_prol = df["Prolongation"] >= hard_thresh
    is_sound = df["SoundRep"] >= hard_thresh
    is_word = df["WordRep"] >= hard_thresh
    is_interj = df["Interjection"] >= hard_thresh
    
    # All positives for target classes
    stutter_positives = df[is_block | is_prol | is_sound | is_word | is_interj].copy()
    fluent_only = df[~(is_block | is_prol | is_sound | is_word | is_interj)].copy()
    
    pos_count = len(stutter_positives)
    needed_fluent = max(1000, subset_size - pos_count)
    
    if len(fluent_only) > needed_fluent:
        sampled_fluent = fluent_only.sample(n=needed_fluent, random_state=seed)
    else:
        sampled_fluent = fluent_only
        
    subset_df = pd.concat([stutter_positives, sampled_fluent]).drop_duplicates(subset=["clip_uid"]).reset_index(drop=True)
    return subset_df

def add_group_kfold_splits(df, n_folds=5, seed=42):
    gkf = GroupKFold(n_splits=n_folds)
    df["fold"] = -1
    groups = df["episode_id"].values
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        df.loc[val_idx, "fold"] = fold
        
    return df

def add_cross_show_splits(df):
    """Fallback split for Exp C: Cross-show split."""
    shows = sorted(df["Show"].unique())
    held_out_shows = ["HVSA", "MyStutteringLife"]
    df["cross_show_split"] = df["Show"].apply(lambda s: "test" if s in held_out_shows else "train")
    return df

def prepare_dataset(config_path="icassp/config.yaml", subset_csv=None, seed=42):
    config = load_config(config_path)
    cache_dir = config["paths"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    
    if subset_csv is None:
        subset_csv = os.path.join(cache_dir, "sep28k_subset.csv")
        
    if os.path.exists(subset_csv):
        print(f"[prep] Loading existing subset from {subset_csv}")
        return pd.read_csv(subset_csv)
        
    print("[prep] Filtering SEP-28k and creating working subset...")
    raw_df = load_and_filter_sep28k(config)
    subset_df = build_working_subset(raw_df, config, subset_size=8000, seed=seed)
    subset_df = add_group_kfold_splits(subset_df, n_folds=config.get("n_folds", 5), seed=seed)
    subset_df = add_cross_show_splits(subset_df)
    
    subset_df.to_csv(subset_csv, index=False)
    print(f"[prep] Saved working subset with {len(subset_df)} clips to {subset_csv}")
    return subset_df

if __name__ == "__main__":
    df = prepare_dataset()
    print("Subset shape:", df.shape)
    print("Fold distribution:\n", df["fold"].value_counts())
    print("Cross-show distribution:\n", df["cross_show_split"].value_counts())
