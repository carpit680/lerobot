#!/usr/bin/env python3
import streamlit as st
from pathlib import Path
import os
import cv2
import numpy as np
from albumentations import Compose, RandomBrightnessContrast
import torch
from huggingface_hub import HfApi
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from augment import run_augmentation

# --- Streamlit UI ---
st.set_page_config(page_title='Lerobot Dataset Augmentation', layout='wide')
st.title('🤖 Lerobot Dataset Augmentation GUI')

# Sidebar for dataset parameters
st.sidebar.header('📦 Dataset Parameters')
repo = st.sidebar.text_input('HF Dataset Repo (user/ds)', value='carpit680/giraffe_clean_desk')
default_token = os.getenv('HF_TOKEN', '')
token = st.sidebar.text_input('HF Access Token', type='password', value=default_token)
cache_dir = st.sidebar.text_input('Cache Directory', value=str(Path.home()/'.cache'/'lerobot'))
max_eps = st.sidebar.number_input('Max Episodes to Process', min_value=1, step=1, value=1)
out_repo = st.sidebar.text_input('Output HF Repo (optional)', value='')
delete_existing = st.sidebar.checkbox('Delete existing HF dataset before push')

st.sidebar.header('🎨 Augmentation Options')
light = st.sidebar.checkbox('Brightness/Contrast Jitter')
seg_color = st.sidebar.checkbox('Segment & Random-Color Robot')
env_swap_opt = st.sidebar.checkbox('Segment & Swap Background')
bg_dir = None
if env_swap_opt:
    bg_dir = st.sidebar.text_input('Background Images Directory Path')
color_alpha = st.sidebar.slider('Color Overlay Alpha', 0.0, 1.0, 0.5, step=0.05)

st.sidebar.header('🛠️ Controls')
if 'stop' not in st.session_state:
    st.session_state.stop = False
if st.sidebar.button('⏹️ Stop Augmentation'):
    st.session_state.stop = True
if st.sidebar.button('🔄 Reset'):
    st.session_state.stop = False
    st.experimental_rerun()

# Preload metadata and frame counts
meta_root = Path(cache_dir) / repo
if meta_root.exists():
    ds_meta = LeRobotDatasetMetadata(repo_id=repo, root=str(meta_root), local_files_only=True)
    cam_key = ds_meta.camera_keys[0]
    video_counts = {}
    for ep in range(int(max_eps)):
        video_path = meta_root / ds_meta.get_video_file_path(ep, cam_key)
        cap = cv2.VideoCapture(str(video_path))
        video_counts[ep] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
else:
    ds_meta = None
    cam_key = None
    video_counts = {}

# Prepare mask generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mask_gen = None
if seg_color or env_swap_opt:
    sam_model = build_sam2(
        'configs/sam2.1/sam2.1_hiera_t.yaml',
        '/mnt/data/Projects/sam2/checkpoints/sam2.1_hiera_tiny.pt',
        device=device,
        apply_postprocessing=False
    )
    mask_gen = SAM2AutomaticMaskGenerator(sam_model)

# Live display placeholders
st.subheader("🔍 Live Frame Preview")
col1, col2 = st.columns(2)
orig_ph = col1.empty()
aug_ph  = col2.empty()
log_box = st.empty()
log_lines = []

# Helpers
def check_stop():
    if st.session_state.stop:
        raise StopIteration("Augmentation stopped by user")

# Run augmentation
st.header('🚀 Run Full Dataset Augmentation')
if st.button('▶️ Run Augmentation'):
    st.session_state.stop = False

    if delete_existing and out_repo.strip():
        try:
            HfApi().delete_repo(repo_id=out_repo.strip(), repo_type='dataset', token=token)
            st.success(f'Deleted existing dataset: {out_repo.strip()}')
        except Exception as e:
            st.error(f'Failed to delete existing dataset: {e}')

    st.text("🔁 Dataset Augmentation Progress")
    dataset_progress = st.progress(0)
    st.text("🎞️ Current Episode Frame Progress")
    episode_progress = st.progress(0)

    def log_cb(line: str):
        log_lines.append(line)
        log_box.code("\n".join(log_lines[-30:]), language="log")

    def progress_cb(done: int, total: int):
        check_stop()
        dataset_progress.progress(done / total)
        episode_progress.progress(0)

    def frame_cb(orig: np.ndarray, aug_rgb: np.ndarray, ep: int, idx: int):
        check_stop()
        orig_ph.image(orig, caption=f'E{ep} ▶️ Orig F{idx}', use_container_width=True)
        aug_ph.image(aug_rgb,  caption=f'E{ep} ▶️ Aug F{idx}', use_container_width=True)
        total = video_counts.get(ep, 1)
        episode_progress.progress(min(idx + 1, total) / total)

    try:
        run_augmentation(
            repo=repo,
            token=token,
            config='configs/sam2.1/sam2.1_hiera_t.yaml',
            checkpoint='/mnt/data/Projects/sam2/checkpoints/sam2.1_hiera_tiny.pt',
            cache_dir=cache_dir,
            max_episodes=int(max_eps),
            out_repo=out_repo.strip() or None,
            light=light,
            seg_color=seg_color,
            env_swap_opt=env_swap_opt,
            bg_dir=bg_dir,
            alpha=color_alpha,
            progress_cb=progress_cb,
            log_cb=log_cb,
            frame_cb=frame_cb,
        )
        st.balloons()
        st.success("🎉 Augmentation complete!")
    except StopIteration:
        st.warning("⏸️ Augmentation stopped by user.")
    except Exception as e:
        st.error(f"❌ Error during augmentation: {e}")
