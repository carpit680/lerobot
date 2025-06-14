#!/usr/bin/env python3
"""
Simple test script to load SAM2 Tiny model and generate masks on an input image.
Usage:
    python test_sam2_tiny.py \
      --config sam2.1/sam2.1_tiny.yaml \
      --checkpoint /path/to/sam2_tiny.pt \
      --image /path/to/image.jpg \
      --output output.png
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_mask_generator(config_path: str, checkpoint_path: str, device: torch.device):
    # build model
    sam = build_sam2(
        config_path,
        checkpoint_path,
        device=device,
        apply_postprocessing=False
    )
    return SAM2AutomaticMaskGenerator(sam)


def overlay_masks(img: np.ndarray, masks: list) -> np.ndarray:
    # overlay each mask with a random color
    overlay = img.copy()
    h, w = img.shape[:2]
    for m in masks:
        seg = m['segmentation'].astype(np.uint8)
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        color = np.random.randint(0, 256, size=(3,), dtype=np.uint8).tolist()
        # apply color to overlay
        overlay[seg > 0] = (overlay[seg>0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Test SAM2 Tiny model mask generation")
    parser.add_argument("--config", required=True, help="Path or package config for SAM2 (e.g. sam2.1/sam2.1_tiny.yaml)")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM2 .pt checkpoint")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output overlay image path")
    args = parser.parse_args()

    # verify inputs
    img_path = Path(args.image)
    assert img_path.is_file(), f"Image not found: {args.image}"
    ckpt = Path(args.checkpoint)
    assert ckpt.is_file(), f"Checkpoint not found: {args.checkpoint}"

    # load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image: {img_path}")
        sys.exit(1)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load mask generator
    print("Loading SAM2 Tiny model...")
    mask_gen = load_mask_generator(args.config, args.checkpoint, device)

    # generate masks
    print("Generating masks...")
    masks = mask_gen.generate(img)
    print(f"Number of masks generated: {len(masks)}")

    # overlay
    overlay = overlay_masks(img, masks)

    # save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"Overlay saved to {out_path}")

if __name__ == '__main__':
    main()
