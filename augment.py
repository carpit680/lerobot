# augment.py

#!/usr/bin/env python3
import os, random, shutil, subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2, numpy as np, torch
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
    h, s, v = np.random.randint(0,180), np.random.randint(50,256), np.random.randint(50,256)
    new_col = np.array([h,s,v],dtype=np.float32)
    mask_bool = mask.astype(bool)
    hsv[mask_bool] = hsv[mask_bool]*(1-alpha) + new_col*alpha
    return cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_individual_colors(img, masks, alpha):
    out = img.copy()
    for m in masks:
        mask = m['segmentation'].astype(np.uint8)
        out = color_replace(out, mask, alpha)
    return out

def env_swap(img, mask, bg_dir):
    bg = cv2.imread(str(random.choice(list(Path(bg_dir).glob('*')))))
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
    fg = img.copy(); fg[mask==0]=0
    inv=mask==0; bg[~inv]=0
    return fg+bg

def encode_with_ffmpeg(img_dir: Path, output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True,exist_ok=True)
    cmd = [
        'ffmpeg','-f','image2','-r',str(fps),
        '-i',str(img_dir/'frame_%06d.png'),
        '-vcodec','libx264','-pix_fmt','yuv420p',
        '-crf','23','-loglevel','error','-y',str(output_path)
    ]
    subprocess.run(cmd,check=True)

def run_augmentation(
    repo: str,
    token: str,
    config: str,
    checkpoint: str,
    cache_dir: Optional[str] = None,
    max_episodes: Optional[int] = None,
    out_repo: Optional[str] = None,
    # segmentation + color
    seg_color: bool = False,
    env_swap_opt: bool = False,
    bg_dir: Optional[str] = None,
    alpha: float = 0.5,
    # brightness/contrast
    light: bool = False,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    # other aug flags
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
    # joint/action/state bump
    joint_aug: bool = False,
    joint_max: float = 0.0,
    # callbacks
    progress_cb: Optional[Callable[[int,int],None]] = None,
    log_cb:      Optional[Callable[[str],None]]      = None,
    frame_cb:    Optional[Callable[...,None]]        = None,
):
    os.environ['HF_HUB_TOKEN'] = token
    base      = Path(cache_dir) if cache_dir else Path.home()/'.cache'/'lerobot'
    meta_root = base/repo; meta_root.mkdir(parents=True,exist_ok=True)

    if log_cb: log_cb(f'Downloading metadata from {repo}…')
    ds_meta = LeRobotDatasetMetadata(repo_id=repo,
                                     root=str(meta_root),
                                     local_files_only=False)
    LeRobotDataset(repo_id=repo,
                   root=str(meta_root),
                   local_files_only=False,
                   download_videos=True)

    # pick out all numeric vectors
    skip_keys   = {'index','frame_index','episode_index','task_index','timestamp'}
    vector_keys = [
        k for k,ft in ds_meta.features.items()
        if ft['dtype'] not in ['image','video'] and k not in skip_keys
    ]
    # any vector that mentions joint, action or state
    aug_keys    = [k for k in vector_keys
                   if any(x in k for x in ('action','observation.state'))]
    total_eps = ds_meta.total_episodes
    num_eps   = min(max_episodes or total_eps, total_eps)
    if log_cb: log_cb(f'Processing {num_eps}/{total_eps} episodes')

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_gen = load_mask_generator(device, config, checkpoint) \
               if (seg_color or env_swap_opt) else None

    out_name = out_repo or f'{repo}-augmented'
    out_root = base/out_name
    if out_root.exists(): shutil.rmtree(out_root)
    new_ds = LeRobotDataset.create(repo_id=out_name,
                                   fps=ds_meta.fps,
                                   features=ds_meta.features,
                                   root=out_root)

    for ep in range(num_eps):
        if log_cb: log_cb(f'▶️ Episode {ep+1}/{num_eps}')
        if progress_cb: progress_cb(ep+1, num_eps)

        parquet    = meta_root/ ds_meta.get_data_file_path(ep)
        hf_ds      = load_dataset('parquet',
                                  data_files=[str(parquet)],
                                  split='train')
        num_frames = len(hf_ds)

        # build per-key sinusoidal offsets
        offsets_map = {}
        if joint_aug and aug_keys and num_frames > 1:
            t      = np.arange(num_frames)
            scales = np.sin(np.pi * t/(num_frames-1))
            for k in aug_keys:
                sample  = hf_ds[0][k]
                n       = len(sample)
                dirs    = np.random.choice([-1,1], size=(n,))
                offsets_map[k] = np.outer(scales, dirs * joint_max)
        else:
            offsets_map = {}

        # open all video captures
        caps = {
            cam: cv2.VideoCapture(str(meta_root/ ds_meta.get_video_file_path(ep,cam)))
            for cam in ds_meta.camera_keys
        }

        for i, row in enumerate(hf_ds):
            orig_dict, aug_dict, frame_data = {}, {}, {}
            # —— image part ——
            for cam,cap in caps.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok: continue

                # build transforms…
                transforms = []
                if light:
                    transforms.append(RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=contrast_limit,p=1))
                if crop:
                    h,w = frame.shape[:2]
                    transforms.append(RandomCrop(
                        height=int(h*crop_frac),
                        width =int(w*crop_frac),p=1))
                if rotate:
                    transforms.append(Rotate(limit=rotate_limit,p=1))
                if hflip:
                    transforms.append(HorizontalFlip(p=hflip_prob))
                if vflip:
                    transforms.append(VerticalFlip(p=vflip_prob))
                if noise:
                    transforms.append(GaussNoise(var_limit=(0,noise_strength),p=1))
                if blur:
                    transforms.append(Blur(blur_limit=blur_limit,p=1))
                if occlusion:
                    h,w = frame.shape[:2]
                    transforms.append(CoarseDropout(
                        max_holes=8,
                        max_height=int(h*occ_size),
                        max_width =int(w*occ_size),p=1))

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
                        combined = np.any(
                            [m['segmentation'] for m in masks], axis=0
                        ).astype(np.uint8)
                        img = env_swap(img, combined, bg_dir)

                orig_dict[cam]  = cv2.cvtColor(frame,   cv2.COLOR_BGR2RGB)
                aug_dict [cam]  = cv2.cvtColor(img,     cv2.COLOR_BGR2RGB)
                frame_data[cam] = cv2.cvtColor(img,     cv2.COLOR_BGR2RGB)

            # —— vector part ——
            orig_v = None
            aug_v  = None
            for k in vector_keys:
                vec = np.array(row[k], dtype=float)
                if k in offsets_map:
                    pert = vec + offsets_map[k][i]
                    frame_data[k] = pert
                else:
                    frame_data[k] = vec

            # pack out one combined vector for charting (we use aug_keys order)
            if joint_aug and aug_keys:
                orig_v = np.concatenate([row[k]           for k in aug_keys])
                aug_v  = np.concatenate([frame_data[k] for k in aug_keys])

            new_ds.add_frame(frame_data)
            if frame_cb:
                frame_cb(orig_dict, aug_dict, ep, i, orig_v, aug_v)

        for cap in caps.values():
            cap.release()
        new_ds.save_episode(task='teleop', encode_videos=False)

    # encode & push
    if log_cb: log_cb('Encoding videos…')
    for ep in range(num_eps):
        for cam in ds_meta.camera_keys:
            img_dir = out_root/'images'/cam/f'episode_{ep:06d}'
            out_vid = out_root/ ds_meta.get_video_file_path(ep,cam)
            encode_with_ffmpeg(img_dir, out_vid, ds_meta.fps)

    if log_cb: log_cb(f'Pushing to {out_name}…')
    new_ds.push_to_hub(private=False, license='apache-2.0')
    if log_cb: log_cb('✅ Done!')
