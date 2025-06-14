#!/usr/bin/env python3
import streamlit as st
from pathlib import Path
import os
import cv2
import numpy as np
from albumentations import Compose, RandomBrightnessContrast
import torch
from huggingface_hub import HfApi

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from augment import run_augmentation

st.set_page_config(page_title='Lerobot Dataset Augmentation', layout='wide')
st.title('Lerobot Dataset Augmentation GUI')

# Sidebar…
repo = st.sidebar.text_input('HF Repo', 'carpit680/giraffe_clean_desk')
token = st.sidebar.text_input('HF Token', type='password', value=os.getenv('HF_TOKEN',''))
cache_dir = st.sidebar.text_input('Cache Dir', str(Path.home()/'.cache'/'lerobot'))
max_eps = st.sidebar.number_input('Max Episodes', 1, step=1, value=1)
out_repo = st.sidebar.text_input('Output Repo (opt)', '')
delete_existing = st.sidebar.checkbox('Delete existing dataset?')

st.sidebar.header('Aug Options')
light     = st.sidebar.checkbox('Brightness/Contrast')
seg_color = st.sidebar.checkbox('Random segment color')
env_swap  = st.sidebar.checkbox('Swap background')
bg_dir    = st.sidebar.text_input('BG images dir') if env_swap else None
alpha     = st.sidebar.slider('Alpha', 0.0, 1.0, 0.5, step=0.05)

# Prepare mask_gen for previews (optional)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mask_gen = None
if seg_color or env_swap:
    sam = build_sam2(
        'configs/sam2.1/sam2.1_hiera_t.yaml',
        '/mnt/data/Projects/sam2/checkpoints/sam2.1_hiera_tiny.pt',
        device=device,
        apply_postprocessing=False
    )
    mask_gen = SAM2AutomaticMaskGenerator(sam)

# placeholders for live frame display
col1, col2 = st.columns(2)
orig_ph = col1.empty()
aug_ph  = col2.empty()

# Run section
if st.button('Run Augmentation'):
    # optionally delete existing…
    if delete_existing:
        try:
            HfApi().delete_repo(repo_id=out_repo or f'{repo}-augmented',
                                repo_type='dataset', token=token)
            st.success("Deleted existing dataset")
        except Exception as e:
            st.error(f"Delete failed: {e}")

    progress = st.progress(0)
    log = st.empty()

    def log_cb(line: str):
        log.text(line)

    def progress_cb(done: int, total: int):
        progress.progress(done / total)

    def frame_cb(orig_bgr: np.ndarray, aug_rgb: np.ndarray, ep: int, idx: int):
        """orig_bgr: BGR frame; aug_rgb: RGB frame"""
        orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_ph.image(orig, caption=f'E{ep} F{idx} ▶️ Orig', use_container_width=True)
        aug_ph.image(aug_rgb, caption=f'E{ep} F{idx} ▶️ Aug',  use_container_width=True)

    run_augmentation(
        repo=repo,
        token=token,
        config='configs/sam2.1/sam2.1_hiera_t.yaml',
        checkpoint='/mnt/data/Projects/sam2/checkpoints/sam2.1_hiera_tiny.pt',
        cache_dir=cache_dir,
        max_episodes=int(max_eps),
        out_repo=out_repo or None,
        light=light,
        seg_color=seg_color,
        env_swap_opt=env_swap,
        bg_dir=bg_dir,
        alpha=alpha,
        progress_cb=progress_cb,
        log_cb=log_cb,
        frame_cb=frame_cb,            # ← hook goes here
    )
    st.success("✅ All done!")
