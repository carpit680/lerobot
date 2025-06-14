#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from albumentations import Compose, RandomBrightnessContrast
from datasets import load_dataset
from huggingface_hub import HfApi
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Lerobot dataset augmentation: lighting, segmentation/color swap, environment swap, and more"
    )
    parser.add_argument("--repo",           required=True, help="HF dataset repo (e.g. user/ds)")
    parser.add_argument("--token",          required=True, help="HF access token")
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "lerobot"),
        help="Base cache dir; per-repo subfolder will be created",
    )
    parser.add_argument("--light",          action="store_true", help="Apply brightness/contrast jitter to frames")
    parser.add_argument("--seg-color",      action="store_true", help="Segment & random-color robot in frames")
    parser.add_argument("--env-swap",       action="store_true", help="Segment & swap backgrounds")
    parser.add_argument("--bg-dir",         help="Directory of background images (for --env-swap)")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (default: all available)",
    )
    parser.add_argument(
        "--out-repo",
        help="Output HF repo (defaults to <repo>-augmented)"
    )
    return parser.parse_args()


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
    h, s, v = random.randint(0,179), random.randint(50,255), random.randint(50,255)
    hsv[mask != 0] = (h, s, v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def env_swap(image, mask, bg_dir):
    bg_file = random.choice(os.listdir(bg_dir))
    bg = cv2.imread(str(Path(bg_dir) / bg_file))
    bg = cv2.resize(bg, (image.shape[1], image.shape[0]))
    fg = image.copy()
    fg[mask == 0] = 0
    inv = (mask == 0)
    bg[~inv] = 0
    return fg + bg


def encode_with_ffmpeg(img_dir: Path, output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-f", "image2", "-r", str(fps),
        "-i", str(img_dir / "frame_%06d.png"),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-loglevel", "error",
        "-y", str(output_path)
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    os.environ["HF_HUB_TOKEN"] = args.token

    cache_base = Path(args.cache_dir)
    repo_cache = cache_base / args.repo
    repo_cache.mkdir(parents=True, exist_ok=True)

    print("🔄 Downloading metadata and videos...")
    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo, root=repo_cache, local_files_only=False)
    LeRobotDataset(repo_id=args.repo, root=repo_cache, local_files_only=False, download_videos=True)

    # Determine episodes to process
    total_eps = ds_meta.total_episodes
    num_eps = args.max_episodes if args.max_episodes and args.max_episodes < total_eps else total_eps
    print(f"🔢 Processing {num_eps}/{total_eps} episodes...")

    # Detect vector keys
    vector_keys = [k for k, ft in ds_meta.features.items() if ft['dtype'] not in ['image','video']]
    skip = {'index','frame_index','episode_index','task_index','timestamp'}
    vector_keys = [k for k in vector_keys if k not in skip]

    # Image augmentation
    img_ops = []
    if args.light:
        img_ops.append(RandomBrightnessContrast(p=1))
    pipeline = Compose(img_ops) if img_ops else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.seg_color or args.env_swap:
        print("🔍 Loading segmentation model...")
        seg_model, preprocess = load_seg_model(device)

    # Prepare output
    out_repo = args.out_repo or f"{args.repo}-augmented"
    out_cache = cache_base / out_repo
    if out_cache.exists():
        shutil.rmtree(out_cache)
    new_ds = LeRobotDataset.create(
        repo_id=out_repo,
        fps=ds_meta.fps,
        features=ds_meta.features,
        root=out_cache
    )
    cam_key = ds_meta.camera_keys[0]

    # Augmentation loops
    for ep in tqdm(range(num_eps), desc='Episodes', unit='ep'):
        parquet_path = repo_cache / ds_meta.get_data_file_path(ep)
        hf_ds = load_dataset('parquet', data_files=[str(parquet_path)], split='train')
        total_frames = len(hf_ds)

        cap = cv2.VideoCapture(str(repo_cache / ds_meta.get_video_file_path(ep, cam_key)))

        for i, row in enumerate(tqdm(hf_ds, desc=f'Ep {ep+1} Frames', total=total_frames, unit='fr')):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            img = pipeline(image=frame)['image'] if pipeline else frame
            if args.seg_color or args.env_swap:
                mask = get_mask(seg_model, preprocess, img, device)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
                if args.seg_color:
                    img = color_replace(img, mask)
                if args.env_swap and args.bg_dir:
                    img = env_swap(img, mask, args.bg_dir)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_data = {cam_key: img_rgb}
            for vk in vector_keys:
                vec = np.array(row[vk])
                frame_data[vk] = vec
            new_ds.add_frame(frame_data)
        cap.release()
        new_ds.save_episode(task='teleop', encode_videos=False)

    # Encode videos if generated
    if num_eps > 0:
        print("🔄 Encoding videos with libx264...")
        for ep in tqdm(range(num_eps), desc='Encoding', unit='ep'):
            img_dir = out_cache / 'images' / cam_key / f"episode_{ep:06d}"
            out_vid = out_cache / ds_meta.get_video_file_path(ep, cam_key)
            encode_with_ffmpeg(img_dir, out_vid, ds_meta.fps)

    print("🚀 Pushing augmented dataset to Hugging Face...")
    new_ds.push_to_hub(private=False, license='apache-2.0')
    print(f"✅ Augmented dataset uploaded: {out_repo}")


if __name__ == '__main__':
    main()


