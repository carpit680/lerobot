#!/usr/bin/env python3
import os
import random
import shutil
import subprocess
import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from albumentations import (
    Compose,
    RandomBrightnessContrast,
    RandomCrop,
    Rotate,
    HorizontalFlip,
    VerticalFlip,
    GaussNoise,
    Blur,
    CoarseDropout,
)

def load_mask_generator(device, config_path, checkpoint_path):
    sam = build_sam2(
        config_path, checkpoint_path,
        device=device, apply_postprocessing=False
    )
    return SAM2AutomaticMaskGenerator(sam)

def color_replace(img, mask, alpha=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = np.random.randint(0, 180), np.random.randint(50, 256), np.random.randint(50, 256)
    new_col = np.array([h, s, v], dtype=np.float32)
    mask_bool = mask.astype(bool)
    hsv[mask_bool] = hsv[mask_bool] * (1 - alpha) + new_col * alpha
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_individual_colors(img, masks, alpha):
    out = img.copy()
    for m in masks:
        mask = m['segmentation'].astype(np.uint8)
        out = color_replace(out, mask, alpha)
    return out

def env_swap(img, mask, bg_dir):
    bg = cv2.imread(str(random.choice(list(Path(bg_dir).glob('*')))))
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
    fg = img.copy(); fg[mask == 0] = 0
    inv = mask == 0; bg[~inv] = 0
    return fg + bg

def encode_with_ffmpeg(img_dir: Path, output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-f', 'image2', '-r', str(fps),
        '-i', str(img_dir / 'frame_%06d.png'),
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '23', '-loglevel', 'error', '-y', str(output_path)
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
    seg_color: bool = False,
    env_swap_opt: bool = False,
    bg_dir: Optional[str] = None,
    alpha: float = 0.5,
    light: bool = False,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    crop: bool = False,
    crop_frac: float = 0.8,
    rotate: bool = False,
    rotate_limit: int = 45,
    hflip: bool = False,
    hflip_prob: float = 0.5,
    vflip: bool = False,
    vflip_prob: float = 0.5,
    noise: bool = False,
    noise_strength: int = 25,
    blur: bool = False,
    blur_limit: int = 7,
    occlusion: bool = False,
    occ_size: float = 0.1,
    joint_aug: bool = False,
    joint_max: float = 0.0,
    progress_cb: Optional[Callable[[int,int],None]] = None,
    log_cb: Optional[Callable[[str],None]] = None,
    frame_cb: Optional[Callable[...,None]] = None,
):
    os.environ['HF_HUB_TOKEN'] = token
    base      = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'lerobot'
    meta_root = base / repo
    meta_root.mkdir(parents=True, exist_ok=True)

    # 1) fetch original metadata + videos
    if log_cb: log_cb(f'Downloading metadata from {repo}…')
    ds_meta = LeRobotDatasetMetadata(repo_id=repo, root=str(meta_root), local_files_only=False)
    LeRobotDataset(repo_id=repo, root=str(meta_root), local_files_only=False, download_videos=True)

    # identify vector keys to augment
    skip_keys   = {'index','frame_index','episode_index','task_index','timestamp'}
    vector_keys = [
        k for k, ft in ds_meta.features.items()
        if ft['dtype'] not in ['image','video'] and k not in skip_keys
    ]
    aug_keys = [k for k in vector_keys if any(x in k for x in ('action','observation.state'))]

    total_eps = ds_meta.total_episodes
    num_eps   = min(max_episodes or total_eps, total_eps)
    if log_cb: log_cb(f'Original episodes: {total_eps}; augmenting: {num_eps}')

    # prepare mask generator if needed
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_gen = load_mask_generator(device, config, checkpoint) if (seg_color or env_swap_opt) else None

    out_name = out_repo or f'{repo}-augmented'
    out_root = base / out_name
    if out_root.exists():
        shutil.rmtree(out_root)
    # fast‐clone (hard‐link) the original cache into your new root:
    if log_cb: log_cb("📁 Cloning original dataset (hard-links) …")
    shutil.copytree(
        meta_root,   # e.g. ~/.cache/lerobot/<repo>
        out_root,    # where your augmented dataset will live
        dirs_exist_ok=True,
        copy_function=os.link
    )
    new_ds = LeRobotDataset(
        repo_id=out_name,
        root=out_root,
        local_files_only=True  # pick up existing data
    )

    # ─── STEP A: Copy all original episodes into the new dataset ────────────────
    for ep in range(total_eps):
        if log_cb: log_cb(f'📀 Copying original episode {ep+1}/{total_eps}')
        parquet = meta_root / ds_meta.get_data_file_path(ep)
        hf_ds   = load_dataset('parquet', data_files=[str(parquet)], split='train')
        caps    = {
            cam: cv2.VideoCapture(str(meta_root / ds_meta.get_video_file_path(ep, cam)))
            for cam in ds_meta.camera_keys
        }

        for i, row in enumerate(hf_ds):
            frame_data = {}
            for cam, cap in caps.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok: continue
                frame_data[cam] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for k in vector_keys:
                frame_data[k] = np.array(row[k], dtype=float)
            new_ds.add_frame(frame_data)

        for cap in caps.values():
            cap.release()
        new_ds.save_episode(task='teleop', encode_videos=False)

    # ─── STEP B: Generate and append the requested augmented episodes ───────────
    new_lengths = []
    for ep in range(num_eps):
        if log_cb: log_cb(f'▶️ Augmenting episode {ep+1}/{num_eps}')
        if progress_cb: progress_cb(ep+1, num_eps)

        parquet = meta_root / ds_meta.get_data_file_path(ep)
        hf_ds   = load_dataset('parquet', data_files=[str(parquet)], split='train')
        num_frames = len(hf_ds)
        new_lengths.append(num_frames)

        # precompute joint‐offsets if needed
        offsets_map = {}
        if joint_aug and aug_keys and num_frames > 1:
            t      = np.arange(num_frames)
            scales = np.sin(np.pi * t/(num_frames-1))
            action_keys = [k for k in aug_keys if 'action' in k]
            obs_keys    = [k for k in aug_keys if 'observation.state' in k]
            if len(action_keys)==1 and len(obs_keys)==1:
                n = len(hf_ds[0][action_keys[0]])
                dirs = np.random.choice([-1,1], size=(n,))
                amps = np.random.uniform(0, joint_max, size=(n,))
                base = np.outer(scales, dirs * amps)
                offsets_map[action_keys[0]] = base
                offsets_map[obs_keys[0]]    = base
            else:
                for k in aug_keys:
                    n = len(hf_ds[0][k])
                    dirs = np.random.choice([-1,1], size=(n,))
                    amps = np.random.uniform(0, joint_max, size=(n,))
                    offsets_map[k] = np.outer(scales, dirs * amps)

        caps = {
            cam: cv2.VideoCapture(str(meta_root / ds_meta.get_video_file_path(ep, cam)))
            for cam in ds_meta.camera_keys
        }

        for i, row in enumerate(hf_ds):
            orig_dict, aug_dict, frame_data = {}, {}, {}
            # image transforms
            for cam, cap in caps.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok: continue

                transforms = []
                if light:
                    transforms.append(RandomBrightnessContrast(
                        brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1))
                if crop:
                    h,w = frame.shape[:2]
                    transforms.append(RandomCrop(
                        height=int(h*crop_frac), width=int(w*crop_frac), p=1))
                if rotate:
                    transforms.append(Rotate(limit=rotate_limit, p=1))
                if hflip:
                    transforms.append(HorizontalFlip(p=hflip_prob))
                if vflip:
                    transforms.append(VerticalFlip(p=vflip_prob))
                if noise:
                    transforms.append(GaussNoise(
                        var_limit=(noise_strength, noise_strength),
                        per_channel=True, p=1))
                if blur:
                    transforms.append(Blur(blur_limit=blur_limit, p=1))
                if occlusion:
                    h,w = frame.shape[:2]
                    transforms.append(CoarseDropout(
                        max_holes=8,
                        max_height=int(h*occ_size),
                        max_width=int(w*occ_size), p=1))

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

                orig_dict[cam]  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                aug_dict [cam]  = cv2.cvtColor(img,   cv2.COLOR_BGR2RGB)
                frame_data[cam] = cv2.cvtColor(img,   cv2.COLOR_BGR2RGB)

            # vector transforms
            orig_v = None
            aug_v  = None
            for k in vector_keys:
                vec = np.array(row[k], dtype=float)
                if k in offsets_map:
                    pert = vec + offsets_map[k][i]
                    frame_data[k] = pert
                else:
                    frame_data[k] = vec

            if joint_aug and aug_keys:
                orig_v = np.concatenate([row[k]           for k in aug_keys])
                aug_v  = np.concatenate([frame_data[k] for k in aug_keys])

            new_ds.add_frame(frame_data)
            if frame_cb:
                frame_cb(orig_dict, aug_dict, ep, i, orig_v, aug_v)

        for cap in caps.values():
            cap.release()
        new_ds.save_episode(task='teleop', encode_videos=False)

    # ─── STEP C: Update meta/episodes.jsonl & meta/info.json ────────────────
    original_eps   = total_eps
    augmented_eps  = num_eps
    new_total_eps  = original_eps + augmented_eps
    sum_new_frames = sum(new_lengths)

    meta_dir  = out_root / 'meta'
    eps_file  = meta_dir / 'episodes.jsonl'
    info_file = meta_dir / 'info.json'

    # Append new episodes
    with open(eps_file, 'a') as f:
        for idx, length in enumerate(new_lengths, start=original_eps):
            entry = {
                "episode_index": idx,
                "tasks": ["teleop"],
                "length": length
            }
            f.write(json.dumps(entry) + "\n")

    # Patch info.json
    with open(info_file, 'r+') as f:
        info = json.load(f)
        info['total_episodes'] = new_total_eps
        info['total_frames']   = info.get('total_frames', 0) + sum_new_frames
        info['total_videos']   = info.get('total_videos', 0) + augmented_eps
        info.setdefault('splits', {})['train'] = f"0:{new_total_eps}"
        f.seek(0)
        json.dump(info, f, indent=2)
        f.truncate()

    # ─── STEP D: Encode videos & push ──────────────────────────────────────────
    if log_cb: log_cb('Encoding videos…')
    for ep in range(new_total_eps):
        for cam in ds_meta.camera_keys:
            img_dir = out_root / 'images' / cam / f'episode_{ep:06d}'
            out_vid = out_root / ds_meta.get_video_file_path(ep, cam)
            encode_with_ffmpeg(img_dir, out_vid, ds_meta.fps)

    if log_cb: log_cb(f'Pushing to {out_name}…')
    new_ds.push_to_hub(private=False, license='apache-2.0')
    if log_cb: log_cb('✅ Done!')
