import os
import subprocess
import cv2
import tempfile
import shutil
from pathlib import Path
import numpy as np

class EgoHOSWrapper:
    def __init__(self, egohos_repo_dir: str):
        self.egohos_repo_dir = Path(egohos_repo_dir)
        self.mmseg_dir = self.egohos_repo_dir / "mmsegmentation"
        
        # Checkpoints and configs
        self.cfg_twohands = "./work_dirs/seg_twohands_ccda/seg_twohands_ccda.py"
        self.ckpt_twohands = "./work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth"
        
        self.cfg_cb = "./work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py"
        self.ckpt_cb = "./work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth"
        
        self.cfg_obj1 = "./work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py"
        self.ckpt_obj1 = "./work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth"
        
        self.cfg_obj2 = "./work_dirs/twohands_cb_to_obj2_ccda/twohands_cb_to_obj2_ccda.py"
        self.ckpt_obj2 = "./work_dirs/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth"
        
        # Verify paths exist
        if not self.mmseg_dir.exists():
            raise ValueError(f"mmsegmentation dir not found in {self.egohos_repo_dir}")

    def infer_video(self, video_path: str, frame_stride: int = 1, max_frames: int = None):
        """
        Extracts frames to a temp directory, runs EgoHOS (all modes), and yields masks per frame.
        """
        input_path = Path(video_path).resolve()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            frames_dir = temp_dir / "images"
            frames_dir.mkdir()
            
            # 1. Extract frames
            cap = cv2.VideoCapture(str(input_path))
            frame_idx = 0
            frame_indices = []
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_stride == 0:
                    cv2.imwrite(str(frames_dir / f"{frame_idx:08d}.jpg"), frame)
                    frame_indices.append(frame_idx)
                frame_idx += 1
            cap.release()
            
            if not frame_indices:
                return
            
            # 2. Run EgoHOS native scripts
            out_twohands = temp_dir / "pred_twohands"
            out_cb = temp_dir / "pred_cb"
            out_obj1 = temp_dir / "pred_obj1"
            out_obj2 = temp_dir / "pred_obj2"
            
            for out in [out_twohands, out_cb, out_obj1, out_obj2]:
                out.mkdir()
                
            # For subprocesses we need to ensure the correct python is used.
            # Use the isolated venv created specifically for EgoHOS to prevent torch version conflicts
            python_exec = str(self.egohos_repo_dir.parent.parent.parent / "venv_egohos" / "bin" / "python")
            if not os.path.exists(python_exec):
                # Fallback for Windows or if running locally without venv_egohos
                python_exec = "python"
                
            # Auto-patch predict_image.py to import torch before mmseg if needed
            predict_py = self.mmseg_dir / "predict_image.py"
            if predict_py.exists():
                content = predict_py.read_text()
                if not content.lstrip().startswith("import torch"):
                    predict_py.write_text("import torch  # Pre-load PyTorch CUDA libraries\n" + content)

            commands = [
                [python_exec, "predict_image.py", "--config_file", self.cfg_twohands, "--checkpoint_file", self.ckpt_twohands, "--img_dir", str(frames_dir), "--pred_seg_dir", str(out_twohands)],
                [python_exec, "predict_image.py", "--config_file", self.cfg_cb, "--checkpoint_file", self.ckpt_cb, "--img_dir", str(frames_dir), "--pred_seg_dir", str(out_cb)],
                [python_exec, "predict_image.py", "--config_file", self.cfg_obj1, "--checkpoint_file", self.ckpt_obj1, "--img_dir", str(frames_dir), "--pred_seg_dir", str(out_obj1)],
                [python_exec, "predict_image.py", "--config_file", self.cfg_obj2, "--checkpoint_file", self.ckpt_obj2, "--img_dir", str(frames_dir), "--pred_seg_dir", str(out_obj2)]
            ]
            
            env = os.environ.copy()
            # Dynamically query torch and nvidia paths from the venv's python
            extra_paths = []
            torch_lib = ""
            
            # 1. Get torch/lib path
            try:
                res_torch = subprocess.run([python_exec, "-c", "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"], capture_output=True, text=True, check=True)
                torch_lib = res_torch.stdout.strip()
                if os.path.isdir(torch_lib):
                    extra_paths.append(torch_lib)
                    
                    # WORKAROUND: If mmcv expects libtorch_cuda_cu.so / libtorch_cuda_cpp.so (PyTorch 1.12) 
                    # but we are on PyTorch 1.13 where they were merged into libtorch_cuda.so, 
                    # we create symlinks to satisfy the loader.
                    cuda_so = os.path.join(torch_lib, "libtorch_cuda.so")
                    if os.path.exists(cuda_so):
                        for lib_name in ["libtorch_cuda_cu.so", "libtorch_cuda_cpp.so"]:
                            missing_so = os.path.join(torch_lib, lib_name)
                            if not os.path.exists(missing_so):
                                try:
                                    os.symlink(cuda_so, missing_so)
                                    print(f"EgoHOSWrapper: Created symlink {missing_so} -> {cuda_so}")
                                except Exception as e:
                                    print(f"EgoHOSWrapper: Failed to create symlink {missing_so}: {e}")
                            
            except Exception as e:
                print(f"Warning: Failed to dynamically query torch path: {e}")
                
            # 2. Get nvidia/lib paths (some pip versions put cuda libs in nvidia package)
            try:
                res_nvidia = subprocess.run([python_exec, "-c", "import os, site; print(site.getsitepackages()[0])"], capture_output=True, text=True, check=True)
                site_packages = res_nvidia.stdout.strip()
                nvidia_dir = os.path.join(site_packages, "nvidia")
                if os.path.isdir(nvidia_dir):
                    for subdir in os.listdir(nvidia_dir):
                        n_lib = os.path.join(nvidia_dir, subdir, "lib")
                        if os.path.isdir(n_lib):
                            extra_paths.append(n_lib)
            except Exception:
                pass
            
            # Print for debugging
            print(f"EgoHOSWrapper: Injected LD_LIBRARY_PATH with: {extra_paths}")
            
            if extra_paths:
                env["LD_LIBRARY_PATH"] = f"{':'.join(extra_paths)}:{env.get('LD_LIBRARY_PATH', '')}"
                
                # Double-check preloading just in case Python hides LD_LIBRARY_PATH
                preload = []
                if torch_lib:
                    for lib in ["libtorch_python.so", "libtorch_cuda_cpp.so", "libtorch_cuda_cu.so", "libtorch_cuda.so", "libc10_cuda.so"]:
                        lpath = os.path.join(torch_lib, lib)
                        if os.path.exists(lpath):
                            preload.append(lpath)
                if preload:
                    env["LD_PRELOAD"] = f"{' '.join(preload)} {env.get('LD_PRELOAD', '')}".strip()
                    print(f"EgoHOSWrapper: Injected LD_PRELOAD with: {preload}")

            for cmd in commands:
                subprocess.run(cmd, cwd=str(self.mmseg_dir), check=True, env=env)
            
            # 3. Yield masks
            for idx in frame_indices:
                filename = f"{idx:08d}.png"
                
                twohands_path = out_twohands / filename
                cb_path = out_cb / filename
                obj1_path = out_obj1 / filename
                obj2_path = out_obj2 / filename
                
                # Load masks (they are grayscale)
                mask_twohands = cv2.imread(str(twohands_path), cv2.IMREAD_GRAYSCALE) if twohands_path.exists() else None
                mask_cb = cv2.imread(str(cb_path), cv2.IMREAD_GRAYSCALE) if cb_path.exists() else None
                mask_obj1 = cv2.imread(str(obj1_path), cv2.IMREAD_GRAYSCALE) if obj1_path.exists() else None
                mask_obj2 = cv2.imread(str(obj2_path), cv2.IMREAD_GRAYSCALE) if obj2_path.exists() else None
                
                # If any missing (shouldn't happen unless inference failed), yield empty
                if mask_twohands is None:
                    continue
                    
                # Parse masks based on EgoHOS class IDs
                # twohands: 1=left, 2=right
                left_hand = (mask_twohands == 1)
                right_hand = (mask_twohands == 2)
                
                # cb: 1=contact boundary
                contact = (mask_cb == 1) if mask_cb is not None else np.zeros_like(left_hand)
                
                # obj1: 3=left obj, 4=right obj (1 and 2 are hands in obj masks)
                left_obj = (mask_obj1 == 3) if mask_obj1 is not None else np.zeros_like(left_hand)
                right_obj = (mask_obj1 == 4) if mask_obj1 is not None else np.zeros_like(left_hand)
                
                # obj2: 3=two interacting obj
                two_obj = (mask_obj2 == 3) if mask_obj2 is not None else np.zeros_like(left_hand)
                
                yield {
                    'frame_idx': idx,
                    'left_hand': left_hand,
                    'right_hand': right_hand,
                    'left_obj': left_obj,
                    'right_obj': right_obj,
                    'two_obj': two_obj,
                    'contact': contact
                }
