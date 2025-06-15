#!/usr/bin/env python3
import streamlit as st
from pathlib import Path
import os
import cv2
from huggingface_hub import HfApi
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import urllib.parse

from augment import run_augmentation

# --- Streamlit UI ---
st.set_page_config(page_title='Lerobot Dataset Augmentation', layout='wide')
st.title('🤖 Lerobot Dataset Augmentation GUI')

# Sidebar: Dataset parameters
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

# Sidebar: all augmentation options together
st.sidebar.header('🎨 Augmentation Options')

light = st.sidebar.checkbox('Brightness/Contrast Jitter')
if light:
    brightness_limit = st.sidebar.slider('  Brightness Limit', 0.0, 1.0, 0.2, step=0.01, key='brightness_limit')
    contrast_limit   = st.sidebar.slider('  Contrast Limit',   0.0, 1.0, 0.2, step=0.01, key='contrast_limit')
else:
    brightness_limit = 0.2
    contrast_limit   = 0.2

seg_color = st.sidebar.checkbox('Segment & Random-Color Robot')
if seg_color:
    color_alpha = st.sidebar.slider('  Color-overlay Alpha', 0.0, 1.0, 0.5, step=0.05, key='color_alpha')
else:
    color_alpha = 0.5
env_swap_opt = st.sidebar.checkbox('Segment & Swap Background')
if env_swap_opt:
    bg_dir = st.sidebar.text_input('  Background Images Directory Path', value="bg")
else:
    bg_dir = None

crop        = st.sidebar.checkbox('Random Crop')
if crop:
    crop_frac     = st.sidebar.slider('  Crop Fraction', 0.1, 1.0, 0.8, step=0.05, key='crop_frac')
else:
    crop_frac = 0.8

rotate      = st.sidebar.checkbox('Rotate')
if rotate:
    rotate_limit  = st.sidebar.slider('  Max Rotation (°)', 0, 180, 45, step=1,  key='rotate_limit')
else:
    rotate_limit = 45

hflip       = st.sidebar.checkbox('Horizontal Flip')
if hflip:
    hflip_prob    = st.sidebar.slider('  Flip Probability', 0.0, 1.0, 0.5, step=0.05, key='hflip_prob')
else:
    hflip_prob = 0.5

vflip       = st.sidebar.checkbox('Vertical Flip')
if vflip:
    vflip_prob    = st.sidebar.slider('  Flip Probability', 0.0, 1.0, 0.5, step=0.05, key='vflip_prob')
else:
    vflip_prob = 0.5

noise       = st.sidebar.checkbox('Gaussian Noise')
if noise:
    noise_strength = st.sidebar.slider('  Noise Strength', 0, 100, 25, step=1, key='noise_strength')
else:
    noise_strength = 25

blur        = st.sidebar.checkbox('Gaussian Blur')
if blur:
    blur_limit    = st.sidebar.slider('  Blur Kernel Size', 1, 50, 7, step=1, key='blur_limit')
else:
    blur_limit = 7

occlusion   = st.sidebar.checkbox('Random Occlusion')
if occlusion:
    occ_size      = st.sidebar.slider('  Occlusion Size Fraction', 0.0, 0.5, 0.1, step=0.01, key='occ_size')
else:
    occ_size = 0.1

# Sidebar: run / stop / reset
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

# Sidebar: visualize link
display_name = out_repo.strip() or f"{repo}-augmented"
path_param   = urllib.parse.quote(f"/{display_name}/episode_0", safe='')
visual_url   = f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={path_param}"
st.sidebar.markdown(
    f"<a href='{visual_url}' target='_blank'>"
    "<button style='width:100%'>🔗 Visualize Dataset</button></a>",
    unsafe_allow_html=True,
)

# Initialize session state
if 'stop' not in st.session_state: st.session_state.stop = False
if 'run'  not in st.session_state: st.session_state.run  = False

# Preload metadata for live preview
meta_root = Path(cache_dir) / repo
if meta_root.exists():
    ds_meta = LeRobotDatasetMetadata(repo_id=repo, root=str(meta_root), local_files_only=True)
    video_counts = {}
    for cam in ds_meta.camera_keys:
        for ep in range(int(max_eps)):
            vid = meta_root / ds_meta.get_video_file_path(ep, cam)
            cap = cv2.VideoCapture(str(vid))
            video_counts[(ep, cam)] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cap.release()
else:
    ds_meta = None
    video_counts = {}

st.subheader("🔍 Live Frame Preview")
frame_placeholders = {}
if ds_meta:
    for cam in ds_meta.camera_keys:
        c1, c2 = st.columns(2)
        frame_placeholders[cam] = (c1.empty(), c2.empty())

log_box, log_lines = st.empty(), []

def check_stop():
    if st.session_state.stop:
        raise StopIteration("Stopped by user")

# When Run is pressed...
if st.session_state.run:
    if delete_existing and out_repo.strip():
        try:
            HfApi().delete_repo(repo_id=out_repo.strip(), repo_type='dataset', token=token)
            st.success(f"Deleted existing dataset: {out_repo.strip()}")
        except Exception as e:
            st.error(f"Failed to delete existing dataset: {e}")

    st.subheader('🚀 Running Augmentation')
    st.text("🔁 Dataset Progress");    dp = st.progress(0)
    st.text("🎞️ Episode Progress"); ep = st.progress(0)

    def log_cb(line):
        log_lines.append(line)
        log_box.code("\n".join(log_lines[-30:]), language="log")

    def progress_cb(done, total):
        check_stop()
        dp.progress(done/total)
        ep.progress(0)

    def frame_cb(o, a, e_i, f_i):
        check_stop()
        for cam, (o_ph, a_ph) in frame_placeholders.items():
            if cam in o and cam in a:
                o_ph.image(o[cam], caption=f"{cam} E{e_i} ▶ Orig F{f_i}", width=320)
                a_ph.image(a[cam], caption=f"{cam} E{e_i} ▶ Aug  F{f_i}", width=320)
                total = video_counts.get((e_i, cam), 1)
                ep.progress(min(f_i+1, total)/total)

    try:
        run_augmentation(
            repo=repo, token=token,
            config='configs/sam2.1/sam2.1_hiera_t.yaml',
            checkpoint='/mnt/data/Projects/sam2/checkpoints/sam2.1_hiera_tiny.pt',
            cache_dir=cache_dir, max_episodes=int(max_eps),
            out_repo=out_repo.strip() or None,
            # segmentation + color
            seg_color=seg_color, env_swap_opt=env_swap_opt, bg_dir=bg_dir, alpha=color_alpha,
            # brightness/contrast
            light=light, brightness_limit=brightness_limit, contrast_limit=contrast_limit,
            # other aug flags + strengths
            crop=crop,       crop_frac=crop_frac,
            rotate=rotate,   rotate_limit=rotate_limit,
            hflip=hflip,     hflip_prob=hflip_prob,
            vflip=vflip,     vflip_prob=vflip_prob,
            noise=noise,     noise_strength=noise_strength,
            blur=blur,       blur_limit=blur_limit,
            occlusion=occlusion, occ_size=occ_size,
            # callbacks
            progress_cb=progress_cb, log_cb=log_cb, frame_cb=frame_cb,
        )
        st.balloons(); st.success("🎉 Augmentation complete!")
    except StopIteration:
        st.warning("⏸️ Augmentation stopped by user.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
