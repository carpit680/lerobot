#!/usr/bin/env python3
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from albumentations import (
    Compose,
    RandomBrightnessContrast,
    RandomCrop,
    Rotate,
    HorizontalFlip,
    VerticalFlip,
    GaussNoise,          # ← corrected
    Blur,
    CoarseDropout,
)

def load_mask_generator(device: torch.device, config_path: str, checkpoint_path: str):
    sam = build_sam2(
        config_path,
        checkpoint_path,
        device=device,
        apply_postprocessing=False
    )
    return SAM2AutomaticMaskGenerator(sam)

def color_replace(image, mask, alpha=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = np.random.randint(0, 180), np.random.randint(50, 256), np.random.randint(50, 256)
    new_color = np.array([h, s, v], dtype=np.float32)
    mask_bool = mask.astype(bool)
    hsv[mask_bool] = hsv[mask_bool] * (1 - alpha) + new_color * alpha
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_individual_colors(image, masks, alpha=0.5):
    out = image.copy()
    for m in masks:
        mask = m['segmentation'].astype(np.uint8)
        out = color_replace(out, mask, alpha)
    return out

def env_swap(image, mask, bg_dir):
    bg_file = random.choice(list(Path(bg_dir).glob('*')))
    bg = cv2.imread(str(bg_file))
    bg = cv2.resize(bg, (image.shape[1], image.shape[0]))
    fg = image.copy()
    fg[mask == 0] = 0
    inv = (mask == 0)
    bg[~inv] = 0
    return fg + bg

def encode_with_ffmpeg(img_dir: Path, output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-f', 'image2', '-r', str(fps),
        '-i', str(img_dir / 'frame_%06d.png'),
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
        '-loglevel', 'error', '-y', str(output_path)
    ]
    subprocess.run(cmd, check=True)

def run_augmentation(
    repo: str,
    token: str,
    config: str,
    checkpoint: str,
    cache_dir: Optional[str] = None,
    max_episodes: Optional[int] = None,
    out_repo: Optional[str] = None,
    light: bool = False,
    seg_color: bool = False,
    env_swap_opt: bool = False,
    bg_dir: Optional[str] = None,
    alpha: float = 0.5,
    # new flags
    crop: bool = False,
    rotate: bool = False,
    hflip: bool = False,
    vflip: bool = False,
    noise: bool = False,
    blur: bool = False,
    occlusion: bool = False,
    progress_cb: Optional[Callable[[int,int],None]] = None,
    log_cb: Optional[Callable[[str],None]] = None,
    frame_cb: Optional[Callable[[dict,dict,int,int],None]] = None,
):
    os.environ['HF_HUB_TOKEN'] = token
    base = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'lerobot'
    meta_root = base / repo
    meta_root.mkdir(parents=True, exist_ok=True)

    if log_cb: log_cb(f'Downloading metadata from {repo}...')
    ds_meta = LeRobotDatasetMetadata(repo_id=repo, root=str(meta_root), local_files_only=False)
    LeRobotDataset(repo_id=repo, root=str(meta_root), local_files_only=False, download_videos=True)

    total_eps = ds_meta.total_episodes
    num_eps = min(max_episodes or total_eps, total_eps)
    if log_cb: log_cb(f'Processing {num_eps}/{total_eps} episodes')

    skip = {'index','frame_index','episode_index','task_index','timestamp'}
    vector_keys = [k for k, ft in ds_meta.features.items()
                   if ft['dtype'] not in ['image','video'] and k not in skip]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_gen = load_mask_generator(device, config, checkpoint) if (seg_color or env_swap_opt) else None

    out_name = out_repo or f'{repo}-augmented'
    out_root = base / out_name
    if out_root.exists():
        shutil.rmtree(out_root)
    new_ds = LeRobotDataset.create(
        repo_id=out_name,
        fps=ds_meta.fps,
        features=ds_meta.features,
        root=out_root
    )

    for ep in range(num_eps):
        if log_cb: log_cb(f'▶️ Episode {ep+1}/{num_eps}')
        if progress_cb: progress_cb(ep+1, num_eps)

        parquet = meta_root / ds_meta.get_data_file_path(ep)
        hf_ds = load_dataset('parquet', data_files=[str(parquet)], split='train')

        caps = {
            cam: cv2.VideoCapture(str(meta_root / ds_meta.get_video_file_path(ep, cam)))
            for cam in ds_meta.camera_keys
        }

        for i, row in enumerate(hf_ds):
            orig_dict, aug_dict = {}, {}
            frame_data = {}

            for cam, cap in caps.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    continue

                # build dynamic albumentations list
                transforms = []
                if light:
                    transforms.append(RandomBrightnessContrast(p=1))
                if crop:
                    h, w = frame.shape[:2]
                    transforms.append(RandomCrop(height=int(h*0.8), width=int(w*0.8), p=1))
                if rotate:
                    transforms.append(Rotate(limit=45, p=1))
                if hflip:
                    transforms.append(HorizontalFlip(p=1))
                if vflip:
                    transforms.append(VerticalFlip(p=1))
                if noise:
                    transforms.append(GaussNoise(var_limit=(10.0,50.0), p=1))
                if blur:
                    transforms.append(Blur(blur_limit=7, p=1))
                if occlusion:
                    h, w = frame.shape[:2]
                    transforms.append(CoarseDropout(
                        max_holes=8,
                        max_height=int(h*0.1),
                        max_width=int(w*0.1),
                        p=1
                    ))

                aug_frame = frame
                if transforms:
                    aug_frame = Compose(transforms)(image=frame)['image']

                img = aug_frame
                if mask_gen:
                    torch.cuda.empty_cache()
                    masks = mask_gen.generate(img)
                    if seg_color:
                        img = apply_individual_colors(img, masks, alpha)
                    if env_swap_opt and bg_dir:
                        combined = np.any([m['segmentation'] for m in masks], axis=0).astype(np.uint8)
                        img = env_swap(img, combined, bg_dir)

                orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                aug_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_dict[cam] = orig_rgb
                aug_dict[cam] = aug_rgb
                frame_data[cam] = aug_rgb

            # copy over vector data
            for k in vector_keys:
                frame_data[k] = np.array(row[k])
            new_ds.add_frame(frame_data)

            if frame_cb:
                frame_cb(orig_dict, aug_dict, ep, i)

        for cap in caps.values():
            cap.release()

        new_ds.save_episode(task='teleop', encode_videos=False)

    if log_cb: log_cb('Encoding videos...')
    for ep in range(num_eps):
        for cam_key in ds_meta.camera_keys:
            img_dir = out_root / 'images' / cam_key / f'episode_{ep:06d}'
            out_vid = out_root / ds_meta.get_video_file_path(ep, cam_key)
            encode_with_ffmpeg(img_dir, out_vid, ds_meta.fps)

    if log_cb: log_cb(f'Pushing to {out_name}…')
    new_ds.push_to_hub(private=False, license='apache-2.0')
    if log_cb: log_cb('✅ Done!')
