import streamlit as st
import subprocess
import sys
from pathlib import Path
import os
import cv2
import numpy as np
from albumentations import Compose, RandomBrightnessContrast
import torch
from huggingface_hub import HfApi

# SAM2 Automatic Mask Generator imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- Helper functions for preview using SAM2 AutomaticMaskGenerator ---
def load_mask_generator(device: torch.device, config_path: str, checkpoint_path: str):
    # build model
    sam = build_sam2(
        config_path,
        checkpoint_path,
        device=device,
        apply_postprocessing=False
    )
    return SAM2AutomaticMaskGenerator(sam)


def get_combined_mask(mask_generator, image):
    # Generates multiple masks and combine into one
    sam_masks = mask_generator.generate(image)
    # Each entry has 'segmentation' boolean mask. Combine all
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for m in sam_masks:
        combined = np.logical_or(combined, m['segmentation'])
    return combined.astype(np.uint8)


def color_replace(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = np.random.randint(0,180), np.random.randint(50,256), np.random.randint(50,256)
    hsv[mask != 0] = (h, s, v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def env_swap(image, mask, bg_image):
    bg = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
    fg = image.copy()
    fg[mask == 0] = 0
    inv = (mask == 0)
    bg[~inv] = 0
    return fg + bg

# --- Streamlit UI ---
st.set_page_config(page_title="Lerobot Dataset Augmentation", layout="wide")
st.title("Lerobot Dataset Augmentation GUI")

# Sidebar for CLI arguments
st.sidebar.header("Dataset Parameters")
repo = st.sidebar.text_input("HF Dataset Repo (user/ds)", value="carpit680/giraffe_clean_desk")
default_token = os.getenv("HF_TOKEN", "")
token = st.sidebar.text_input("HF Access Token", type="password", value=default_token)
cache_dir = st.sidebar.text_input("Cache Directory", value=str(Path.home()/".cache"/"lerobot"))
max_eps = st.sidebar.number_input("Max Episodes to Process", min_value=1, step=1, value=1)
out_repo = st.sidebar.text_input("Output HF Repo (optional)", value="")
delete_existing = st.sidebar.checkbox("Delete existing HF dataset before push")

# SAM2 config inputs
# st.sidebar.header("SAM2 Mask Generator Config")
# model_cfg = st.sidebar.text_input("SAM2 config YAML path", value="sam2.1/sam2.1_tiny.yaml")
# checkpoint_path = st.sidebar.text_input("SAM2 checkpoint path", value="/mnt/data/Projects/lerobot/sam2.1_hiera_tiny.pt")

st.sidebar.header("Augmentation Options")
light = st.sidebar.checkbox("Brightness/Contrast Jitter")
seg_color = st.sidebar.checkbox("Segment & Random-Color Robot")
env_swap_opt = st.sidebar.checkbox("Segment & Swap Background")
bg_dir = None
if env_swap_opt:
    bg_dir = st.sidebar.text_input("Background Images Directory Path")

# Preview section
st.header("Preview Augmentation on Single Image")
uploaded = st.file_uploader("Upload an image for preview", type=["png", "jpg", "jpeg"])
bg_uploaded = None
if env_swap_opt:
    bg_uploaded = st.file_uploader("Upload a background image", type=["png", "jpg", "jpeg"])

if uploaded:
    np_img = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Lighting
    if light:
        pipeline = Compose([RandomBrightnessContrast(p=1)])
        img = pipeline(image=img)['image']
    # Segmentation with SAM2 AutomaticMaskGenerator
    if seg_color or env_swap_opt:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask_gen = load_mask_generator(device, "configs/sam2.1/sam2.1_hiera_t.yaml", "/mnt/data/Projects/sam2/checkpoints/sam2.1_hiera_tiny.pt")
        mask = get_combined_mask(mask_gen, img)
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
        if seg_color:
            img = color_replace(img, mask)
        if env_swap_opt and bg_uploaded:
            bg_arr = np.frombuffer(bg_uploaded.read(), np.uint8)
            bg_img = cv2.imdecode(bg_arr, cv2.IMREAD_COLOR)
            img = env_swap(img, mask, bg_img)
    aug = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2)
    col1.image(orig, caption="Original", use_container_width=True)
    col2.image(aug, caption="Augmented", use_container_width=True)

# Full run section
st.header("Run Full Dataset Augmentation")
if st.button("Run Augmentation"):
    target_repo = out_repo.strip() or f"{repo}-augmented"
    api = HfApi()
    if delete_existing:
        try:
            api.delete_repo(repo_id=target_repo, repo_type='dataset', token=token)
            st.success(f"Deleted existing dataset: {target_repo}")
        except Exception as e:
            st.error(f"Failed to delete existing dataset: {e}")
    cmd = [sys.executable, "augment.py", "--repo", repo, "--token", token]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    if max_eps > 0:
        cmd += ["--max-episodes", str(int(max_eps))]
    if out_repo:
        cmd += ["--out-repo", out_repo]
    if light:
        cmd.append("--light")
    if seg_color:
        cmd.append("--seg-color")
    if env_swap_opt and bg_dir:
        cmd += ["--env-swap", "--bg-dir", bg_dir]
    st.text("Running: {}".format(" ".join(cmd)))
    with st.spinner("Augmenting dataset, this may take a while..."):
        result = subprocess.run(cmd, capture_output=True, text=True)
    st.subheader("Logs")
    st.text_area("Output", result.stdout + "\n" + result.stderr, height=400)

st.markdown("---")
st.markdown("**Usage:** Place this file alongside your `augment.py` script. Then run `streamlit run streamlit_app.py`.")
