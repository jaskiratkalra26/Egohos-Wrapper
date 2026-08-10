import os
import sys
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np

# mmseg imports
from mmseg.apis import init_segmentor, inference_segmentor

def parse_args():
    parser = argparse.ArgumentParser(description="Run EgoHOS multi-model inference on a folder of frames")
    parser.add_argument("--img_dir", required=True, help="Directory containing frame images")
    parser.add_argument("--cfg_twohands", required=True)
    parser.add_argument("--ckpt_twohands", required=True)
    parser.add_argument("--out_twohands", required=True)

    parser.add_argument("--cfg_cb", required=True)
    parser.add_argument("--ckpt_cb", required=True)
    parser.add_argument("--out_cb", required=True)

    parser.add_argument("--cfg_obj1", required=True)
    parser.add_argument("--ckpt_obj1", required=True)
    parser.add_argument("--out_obj1", required=True)

    parser.add_argument("--cfg_obj2", required=True)
    parser.add_argument("--ckpt_obj2", required=True)
    parser.add_argument("--out_obj2", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    img_paths = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
    if not img_paths:
        print(f"No image files found in {img_dir}")
        return

    device = "cuda:0"

    print("[EgoHOS predict_all] Initializing models...")
    model_twohands = init_segmentor(args.cfg_twohands, args.ckpt_twohands, device=device)
    model_cb = init_segmentor(args.cfg_cb, args.ckpt_cb, device=device)
    model_obj1 = init_segmentor(args.cfg_obj1, args.ckpt_obj1, device=device)
    model_obj2 = init_segmentor(args.cfg_obj2, args.ckpt_obj2, device=device)

    out_dirs = {
        "twohands": (model_twohands, Path(args.out_twohands)),
        "cb": (model_cb, Path(args.out_cb)),
        "obj1": (model_obj1, Path(args.out_obj1)),
        "obj2": (model_obj2, Path(args.out_obj2)),
    }

    for name, (_, p) in out_dirs.items():
        p.mkdir(parents=True, exist_ok=True)

    print(f"[EgoHOS predict_all] Pass 1: Generating twohands masks for {len(img_paths)} frames...")
    for img_path in img_paths:
        fname = Path(img_path).stem + ".png"
        result = inference_segmentor(model_twohands, img_path)
        mask = result[0].astype(np.uint8)
        cv2.imwrite(str(Path(args.out_twohands) / fname), mask)

    print(f"[EgoHOS predict_all] Pass 2: Generating cb, obj1, and obj2 masks for {len(img_paths)} frames...")
    for img_path in img_paths:
        fname = Path(img_path).stem + ".png"
        
        res_cb = inference_segmentor(model_cb, img_path)
        cv2.imwrite(str(Path(args.out_cb) / fname), res_cb[0].astype(np.uint8))

        res_obj1 = inference_segmentor(model_obj1, img_path)
        cv2.imwrite(str(Path(args.out_obj1) / fname), res_obj1[0].astype(np.uint8))

        res_obj2 = inference_segmentor(model_obj2, img_path)
        cv2.imwrite(str(Path(args.out_obj2) / fname), res_obj2[0].astype(np.uint8))

    print("[EgoHOS predict_all] Done!")

if __name__ == "__main__":
    main()
