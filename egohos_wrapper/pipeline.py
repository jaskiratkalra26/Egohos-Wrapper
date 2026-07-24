import os
import subprocess
import cv2
from pathlib import Path
from .postprocessor import EgoHOSPostProcessor

class EgoHOSPipeline:
    def __init__(
        self, 
        egohos_repo_dir: str, 
        output_dir: str = "./egohos_outputs",
        post_processor_kwargs: dict = None
    ):
        """
        Initializes the EgoHOS VM Pipeline wrapper.
        """
        self.egohos_repo_dir = Path(egohos_repo_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if post_processor_kwargs is None:
            post_processor_kwargs = {}
        self.post_processor = EgoHOSPostProcessor(**post_processor_kwargs)

    def run_inference_video(self, input_video_path: str, mode: str = "obj1", frame_stride: int = 1):
        """
        Processes a video through EgoHOS. If frame_stride > 1, it will manually extract 
        every Nth frame and pass the frames to EgoHOS, massively speeding up inference.
        """
        input_path = Path(input_video_path).resolve()
        video_name = input_path.stem
        
        # 1. Extract frames with stride
        frames_dir = self.output_dir / video_name / "images"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting frames from {input_path} (stride={frame_stride})...")
        cap = cv2.VideoCapture(str(input_path))
        frame_idx = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_stride == 0:
                frame_path = frames_dir / f"{frame_idx:08d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
                
            frame_idx += 1
        cap.release()
        print(f"Extracted {extracted_count} frames.")

        # 2. Run EgoHOS native scripts on the extracted frames folder!
        # EgoHOS chains multiple models together: twohands -> cb -> obj1
        mmseg_dir = self.egohos_repo_dir / "mmsegmentation"
        
        # Paths to configs (using default EgoHOS paths)
        cfg_twohands = "./work_dirs/seg_twohands_ccda/seg_twohands_ccda.py"
        ckpt_twohands = "./work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth"
        out_twohands = self.output_dir / video_name / "pred_twohands"
        out_twohands.mkdir(parents=True, exist_ok=True)
        
        cfg_cb = "./work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py"
        ckpt_cb = "./work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth"
        out_cb = self.output_dir / video_name / "pred_cb"
        out_cb.mkdir(parents=True, exist_ok=True)
        
        cfg_obj = f"./work_dirs/twohands_cb_to_{mode}_ccda/twohands_cb_to_{mode}_ccda.py"
        ckpt_obj = f"./work_dirs/twohands_cb_to_{mode}_ccda/best_mIoU_iter_34000.pth" if mode == "obj1" else f"./work_dirs/twohands_cb_to_{mode}_ccda/best_mIoU_iter_32000.pth"
        out_obj = self.output_dir / video_name / f"pred_{mode}"
        out_obj.mkdir(parents=True, exist_ok=True)

        commands = [
            ["python", "predict_image.py", "--config_file", cfg_twohands, "--checkpoint_file", ckpt_twohands, "--img_dir", str(frames_dir), "--pred_seg_dir", str(out_twohands)],
            ["python", "predict_image.py", "--config_file", cfg_cb, "--checkpoint_file", ckpt_cb, "--img_dir", str(frames_dir), "--pred_seg_dir", str(out_cb)],
            ["python", "predict_image.py", "--config_file", cfg_obj, "--checkpoint_file", ckpt_obj, "--img_dir", str(frames_dir), "--pred_seg_dir", str(out_obj)]
        ]
        
        try:
            for i, cmd in enumerate(commands):
                print(f"Running EgoHOS step {i+1}/3...")
                subprocess.run(cmd, cwd=str(mmseg_dir), check=True)
            print(f"EgoHOS inference complete! Raw masks saved to: {out_obj}")
            return str(out_obj)
        except subprocess.CalledProcessError as e:
            print(f"Error during EgoHOS execution: {e}")
            raise e

    def apply_post_processing(self, mask_images_dir: str):
        """
        Iterates over the generated mask frames from EgoHOS and cleans them up using
        the PostProcessor to eliminate mask leakages.
        """
        import numpy as np
        
        mask_dir = Path(mask_images_dir)
        print(f"Applying post-processing to masks in: {mask_dir}")
        
        for img_path in mask_dir.glob("*.png"):
            mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            hand_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
            obj_mask = ((mask == 3) | (mask == 4)).astype(np.uint8)
            
            clean_obj_mask = self.post_processor.process_frame(hand_mask, obj_mask)
            
            final_mask = mask.copy()
            final_mask[(mask == 3) | (mask == 4)] = 0
            final_mask[clean_obj_mask > 0] = 3
            
            cv2.imwrite(str(img_path), final_mask)
            
        print("Post-processing complete. Masks are now leakage-free!")
