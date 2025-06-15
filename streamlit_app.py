# streamlit_app.py

#!/usr/bin/env python3
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2
from huggingface_hub import HfApi
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import urllib.parse

from augment import run_augmentation

UPDATE_EVERY = 5

# --- Streamlit UI ---
st.set_page_config(page_title='Lerobot Dataset Augmentation', layout='wide')
st.title('🤖 Lerobot Dataset Augmentation GUI')

# ─── Sidebar: Dataset parameters ──────────────────────────────────────────────
st.sidebar.header('📦 Dataset Parameters')
token        = st.sidebar.text_input('HF Access Token', type='password', value=os.getenv('HF_TOKEN',''))
hf_api       = HfApi(token=token or None)
username     = st.sidebar.text_input('HuggingFace Username', value='carpit680')
available_repos = []
if username:
    try:
        available_repos = [r.id for r in hf_api.list_datasets(author=username)]
    except Exception as e:
        st.sidebar.error(f"Error fetching repos: {e}")
repo         = st.sidebar.selectbox('Select Dataset Repo', available_repos) if available_repos else ''
cache_dir    = st.sidebar.text_input('Cache Directory', value=str(Path.home()/'.cache'/'lerobot'))
max_eps      = st.sidebar.number_input('Max Episodes to Process', min_value=1, step=1, value=1)
out_repo     = st.sidebar.text_input('Output HF Repo (optional)', value='')
delete_existing = st.sidebar.checkbox('Delete existing HF dataset before push')

# ─── Sidebar: Augmentation Options ────────────────────────────────────────────
st.sidebar.header('🎨 Augmentation Options')

light        = st.sidebar.checkbox('Brightness/Contrast Jitter')
brightness_limit, contrast_limit = (
    st.sidebar.slider('  Brightness Limit', 0.0, 1.0, 0.2, step=0.01),
    st.sidebar.slider('  Contrast Limit',   0.0, 1.0, 0.2, step=0.01),
) if light else (0.2, 0.2)

seg_color    = st.sidebar.checkbox('Segment & Random-Color Robot')
color_alpha  = st.sidebar.slider('  Color-overlay Alpha', 0.0, 1.0, 0.5, step=0.05) if seg_color else 0.5

env_swap_opt = st.sidebar.checkbox('Segment & Swap Background')
bg_dir       = st.sidebar.text_input('  Background Images Directory Path', value="bg") if env_swap_opt else None

crop         = st.sidebar.checkbox('Random Crop')
crop_frac    = st.sidebar.slider('  Crop Fraction', 0.1, 1.0, 0.8, step=0.05) if crop else 0.8

rotate       = st.sidebar.checkbox('Rotate')
rotate_limit = st.sidebar.slider('  Max Rotation (°)', 0, 180, 45, step=1) if rotate else 45

hflip        = st.sidebar.checkbox('Horizontal Flip')
hflip_prob   = st.sidebar.slider('  Flip Probability', 0.0, 1.0, 0.5, step=0.05) if hflip else 0.5

vflip        = st.sidebar.checkbox('Vertical Flip')
vflip_prob   = st.sidebar.slider('  Flip Probability', 0.0, 1.0, 0.5, step=0.05) if vflip else 0.5

noise        = st.sidebar.checkbox('Gaussian Noise')
noise_strength = st.sidebar.slider('  Noise Strength', 0, 100, 25, step=1) if noise else 25

blur         = st.sidebar.checkbox('Gaussian Blur')
blur_limit   = st.sidebar.slider('  Blur Kernel Size', 1, 50, 7, step=1) if blur else 7

occlusion    = st.sidebar.checkbox('Random Occlusion')
occ_size     = st.sidebar.slider('  Occlusion Size Fraction', 0.0, 0.5, 0.1, step=0.01) if occlusion else 0.1

# ─── Joint-Trajectory Augmentation ───────────────────────────────────────────
joint_aug = st.sidebar.checkbox('Joint Trajectory Augmentation')
if joint_aug:
    # slider in degrees, 0–10°
    max_deg = st.sidebar.slider(
        '  Max Joint Variation (°)',
        0.0, 10.0, 5.0, step=0.1, key='joint_max_deg'
    )
    # convert to radians for internal use
    joint_max = max_deg
else:
    joint_max = 0.0
# ─── Sidebar: Run / Stop / Reset ─────────────────────────────────────────────
st.sidebar.header('🛠️ Controls')
c1, c2, c3 = st.sidebar.columns(3)
if c1.button('▶️ Run'):
    st.session_state.stop = False
    st.session_state.run  = True
if c2.button('⏹️ Stop'):
    st.session_state.stop = True
if c3.button('🔄 Reset'):
    st.session_state.stop = False
    st.session_state.run  = False
    for k in ('orig_vals','aug_vals'):
        st.session_state.pop(k, None)
    st.rerun()

# ─── Sidebar: Visualize Link ──────────────────────────────────────────────────
display_name = out_repo.strip() or f"{repo}-augmented"
path_param   = urllib.parse.quote(f"/{display_name}/episode_0", safe='')
visual_url   = f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={path_param}"
st.sidebar.markdown(
    f"<a href='{visual_url}' target='_blank'>"
    "<button style='width:100%'>🔗 Visualize Dataset</button></a>",
    unsafe_allow_html=True,
)

# initialize session flags
st.session_state.setdefault('stop', False)
st.session_state.setdefault('run',  False)

# ─── Preload metadata & compute keys ─────────────────────────────────────────
meta_root = Path(cache_dir) / repo
if meta_root.exists():
    ds_meta = LeRobotDatasetMetadata(repo_id=repo,
                                     root=str(meta_root),
                                     local_files_only=True)
    video_counts = {
        (ep,cam): int(cv2.VideoCapture(str(meta_root/ ds_meta.get_video_file_path(ep,cam)))
                      .get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for cam in ds_meta.camera_keys
        for ep  in range(int(max_eps))
    }
    skip = {'index','frame_index','episode_index','task_index','timestamp'}
    vector_keys = [
        k for k,ft in ds_meta.features.items()
        if ft['dtype'] not in ['image','video'] and k not in skip
    ]
    # any key containing these substrings will be bumped
    aug_keys   = [k for k in vector_keys
                  if any(x in k for x in ('joint','action','state'))]
else:
    ds_meta = None
    video_counts = {}
    vector_keys = []
    aug_keys    = []

# ─── Live Frame Preview ──────────────────────────────────────────────────────
st.subheader("🔍 Live Frame Preview")
frame_placeholders = {}
if ds_meta:
    for cam in ds_meta.camera_keys:
        col1, col2 = st.columns(2)
        frame_placeholders[cam] = (col1.empty(), col2.empty())

log_box, log_lines = st.empty(), []

def check_stop():
    if st.session_state.stop:
        raise StopIteration("Stopped by user")

# ─── Chart Placeholders ──────────────────────────────────────────────────────
if joint_aug and aug_keys:
    st.subheader("🎛️ Joint Trajectory Preview")
    col_o, col_a = st.columns(2)
    orig_chart_ph = col_o.empty()
    aug_chart_ph  = col_a.empty()
else:
    orig_chart_ph = aug_chart_ph = None

# ─── When Run is pressed ──────────────────────────────────────────────────────
if st.session_state.run:
    # init buffers
    if joint_aug and aug_keys and 'orig_vals' not in st.session_state:
        st.session_state.orig_vals = []
        st.session_state.aug_vals  = []

    # delete old repo?
    if delete_existing and out_repo.strip():
        try:
            HfApi().delete_repo(repo_id=out_repo.strip(),
                                repo_type='dataset', token=token)
            st.success(f"Deleted existing dataset: {out_repo.strip()}")
        except Exception as e:
            st.error(f"Failed to delete existing dataset: {e}")

    st.subheader('🚀 Running Augmentation')
    st.text("🔁 Dataset Progress");  dp = st.progress(0)
    st.text("🎞️ Episode Progress"); ep = st.progress(0)

    def log_cb(line):
        log_lines.append(line)
        log_box.code("\n".join(log_lines[-30:]), language="log")

    def progress_cb(done, total):
        check_stop()
        dp.progress(done/total)
        ep.progress(0)

    def frame_cb(o, a, epi, frm, orig_v, aug_v):
        check_stop()

        # 1) update images
        for cam, (o_ph, a_ph) in frame_placeholders.items():
            if cam in o and cam in a:
                o_ph.image(o[cam], caption=f"{cam} E{epi} ▶ Orig F{frm}", width=320)
                a_ph.image(a[cam], caption=f"{cam} E{epi} ▶ Aug  F{frm}", width=320)

        # 2) if charts are active and we have vector data:
        if not (orig_chart_ph and orig_v is not None):
            return

        # append to session buffers
        st.session_state.orig_vals.append(orig_v)
        st.session_state.aug_vals .append(aug_v)

        # helper to build & draw an Altair chart
        def draw_altair(data_list, placeholder, name):
            # build wide DataFrame
            df = pd.DataFrame(data_list)
            df['frame'] = df.index
            # melt to long
            df_long = df.melt(
                id_vars='frame',
                var_name='variable',
                value_name='value'
            )
            # split first half dims → action, rest → state
            n = df.shape[1] - 1
            half = n // 2
            df_long['type'] = df_long['variable'].astype(int).apply(
                lambda x: 'action' if x < half else 'state'
            )

            chart = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X('frame:Q', title='Time'),
                    y=alt.Y('value:Q', axis=alt.Axis(title='Degrees')),
                    color=alt.Color('variable:N', legend=None),
                    strokeDash=alt.StrokeDash('type:N', legend=alt.Legend(title="Type")),
                )
                .properties(
                    width=350,
                    height=250,
                    title=name
                )
            )

            placeholder.altair_chart(chart, use_container_width=True)

        if (frm % UPDATE_EVERY) == 0:
            draw_altair(st.session_state.orig_vals, orig_chart_ph, name="Original")
            draw_altair(st.session_state.aug_vals,  aug_chart_ph,  name="Augmented")

    try:
        run_augmentation(
            repo=repo, token=token,
            config='configs/sam2.1/sam2.1_hiera_t.yaml',
            checkpoint='/mnt/data/Projects/sam2/checkpoints/sam2.1_hiera_tiny.pt',
            cache_dir=cache_dir, max_episodes=int(max_eps),
            out_repo=out_repo.strip() or None,
            # image‐based
            seg_color=seg_color, env_swap_opt=env_swap_opt,
            bg_dir=bg_dir, alpha=color_alpha,
            light=light, brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            crop=crop, crop_frac=crop_frac,
            rotate=rotate, rotate_limit=rotate_limit,
            hflip=hflip, hflip_prob=hflip_prob,
            vflip=vflip, vflip_prob=vflip_prob,
            noise=noise, noise_strength=noise_strength,
            blur=blur, blur_limit=blur_limit,
            occlusion=occlusion, occ_size=occ_size,
            # joint/action/state bump
            joint_aug=joint_aug, joint_max=joint_max,
            progress_cb=progress_cb, log_cb=log_cb,
            frame_cb=frame_cb,
        )
        st.balloons(); st.success("🎉 Augmentation complete!")
    except StopIteration:
        st.warning("⏸️ Augmentation stopped by user.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
