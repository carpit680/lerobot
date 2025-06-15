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
import urllib.parse

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from augment import run_augmentation

# --- Streamlit UI ---
st.set_page_config(page_title='Lerobot Dataset Augmentation', layout='wide')
st.title('🤖 Lerobot Dataset Augmentation GUI')

# Sidebar for dataset parameters
st.sidebar.header('📦 Dataset Parameters')
default_token = os.getenv('HF_TOKEN', '')
token = st.sidebar.text_input('HF Access Token', type='password', value=default_token)

hf_api = HfApi(token=token if token else None)
username = st.sidebar.text_input('HuggingFace Username', value='carpit680')

available_repos = []
if username:
    try:
        available_repos = [repo.id for repo in hf_api.list_datasets(author=username)]
    except Exception as e:
        st.sidebar.error(f"Error fetching repos: {e}")

repo = st.sidebar.selectbox('Select Dataset Repo', available_repos) if available_repos else ''

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

# Sidebar for controls
st.sidebar.header('🛠️ Controls')
controls = st.sidebar.columns(3)
if controls[0].button('▶️ Run'):
    st.session_state.stop = False
    st.session_state.run = True
if controls[1].button('⏹️ Stop'):
    st.session_state.stop = True
if controls[2].button('🔄 Reset'):
    st.session_state.stop = False
    st.session_state.run = False
    st.rerun()

# Visualize Dataset button
# Determine the output dataset name (default to '<repo>-augmented' if none provided)
display_name = out_repo.strip() if out_repo.strip() else f"{repo}-augmented"
path_param = urllib.parse.quote(f"/{display_name}/episode_0", safe='')
visual_url = f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={path_param}"
st.sidebar.markdown(f"<a href='{visual_url}' target='_blank'><button style='width:100%'>🔗 Visualize Dataset</button></a>", unsafe_allow_html=True)

# Initialize session state
if 'stop' not in st.session_state:
    st.session_state.stop = False
if 'run' not in st.session_state:
    st.session_state.run = False

# Preload metadata and frame counts
meta_root = Path(cache_dir) / repo
if meta_root.exists():
    ds_meta = LeRobotDatasetMetadata(repo_id=repo, root=str(meta_root), local_files_only=True)
    video_counts = {}
    for cam_key in ds_meta.camera_keys:
        for ep in range(int(max_eps)):
            video_path = meta_root / ds_meta.get_video_file_path(ep, cam_key)
            cap = cv2.VideoCapture(str(video_path))
            video_counts[(ep, cam_key)] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cap.release()
else:
    ds_meta = None
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
frame_placeholders = {}
if ds_meta:
    for cam_key in ds_meta.camera_keys:
        cols = st.columns(2)
        frame_placeholders[cam_key] = (cols[0].empty(), cols[1].empty())

log_box = st.empty()
log_lines = []

def check_stop():
    if st.session_state.stop:
        raise StopIteration("Augmentation stopped by user")

# Run augmentation when triggered
if st.session_state.run:
    # Delete existing if requested
    if delete_existing and out_repo.strip():
        try:
            HfApi().delete_repo(repo_id=out_repo.strip(), repo_type='dataset', token=token)
            st.success(f'Deleted existing dataset: {out_repo.strip()}')
        except Exception as e:
            st.error(f'Failed to delete existing dataset: {e}')

    st.subheader('🚀 Running Augmentation')
    st.text("🔁 Overall Dataset Progress")
    dataset_progress = st.progress(0)
    st.text("🎞️ Episode Frame Progress")
    episode_progress = st.progress(0)

    def log_cb(line: str):
        log_lines.append(line)
        log_box.code("\n".join(log_lines[-30:]), language="log")

    def progress_cb(done: int, total: int):
        check_stop()
        dataset_progress.progress(done / total)
        episode_progress.progress(0)

    def frame_cb(orig_dict: dict, aug_dict: dict, ep: int, idx: int):
        check_stop()
        for cam_key, (orig_ph, aug_ph) in frame_placeholders.items():
            if cam_key in orig_dict and cam_key in aug_dict:
                orig_ph.image(orig_dict[cam_key], caption=f'{cam_key} E{ep} ▶️ Orig F{idx}', width=320)
                aug_ph.image(aug_dict[cam_key], caption=f'{cam_key} E{ep} ▶️ Aug F{idx}', width=320)
                total = video_counts.get((ep, cam_key), 1)
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