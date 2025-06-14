import streamlit as st
import subprocess
import sys
from pathlib import Path
import os
import cv2
import numpy as np
from albumentations import Compose, RandomBrightnessContrast
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

# --- Helper functions for preview ---
def load_seg_model(device):
    model = deeplabv3_resnet50(pretrained=True).eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(520),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, preprocess


def get_mask(model, preprocess, image, device):
    inp = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)["out"][0]
    return out.argmax(0).cpu().numpy().astype(np.uint8)


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
repo = st.sidebar.text_input("HF Dataset Repo (user/ds)", value="user/dataset")
token = st.sidebar.text_input("HF Access Token", type="password")
cache_dir = st.sidebar.text_input("Cache Directory", value=str(Path.home()/".cache"/"lerobot"))
max_eps = st.sidebar.number_input("Max Episodes to Process", min_value=0, step=1, value=0)
out_repo = st.sidebar.text_input("Output HF Repo (optional)", value="")

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
    ops = []
    if light:
        pipeline = Compose([RandomBrightnessContrast(p=1)])
        img = pipeline(image=img)['image']
    if seg_color or env_swap_opt:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seg_model, preprocess_fn = load_seg_model(device)
        mask = get_mask(seg_model, preprocess_fn, img, device)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
        if seg_color:
            img = color_replace(img, mask)
        if env_swap_opt and bg_uploaded:
            bg_arr = np.frombuffer(bg_uploaded.read(), np.uint8)
            bg_img = cv2.imdecode(bg_arr, cv2.IMREAD_COLOR)
            img = env_swap(img, mask, bg_img)
    aug = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2)
    col1.image(orig, caption="Original", use_column_width=True)
    col2.image(aug, caption="Augmented", use_column_width=True)

# Full run section
st.header("Run Full Dataset Augmentation")
if st.button("Run Augmentation"):
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
