# EgoHOS Integration Module

This module wraps the EgoHOS segmentation pipeline and provides customizable OpenCV post-processing to eliminate "mask leakage" (where object masks bleed onto identical background items).

## Quick Start (For your VM)

```python
from egohos_wrapper import EgoHOSPipeline

# 1. Initialize the pipeline
pipeline = EgoHOSPipeline(
    egohos_repo_dir="/home/jaskirats2004/EgoHOS", 
    output_dir="/home/jaskirats2004/processed_outputs",
    post_processor_kwargs={
        "max_distance_px": 200,          # How far an object can be from the hand
        "morph_kernel_size": 5,          # Strength of bridge-breaking erosion
        "min_blob_area_px": 150,         # Remove tiny noise blobs
        "use_connected_components": True # Ensure object touches hand
    }
)

# 2. Run EgoHOS inference with frame-skipping (stride)
video_path = "/home/jaskirats2004/EgoHOS/pipeline/data/raw/new_custom_worker/tape_video_30sec.mp4"
pipeline.run_inference_video(video_path, mode="obj1", frame_stride=3) # Processes every 3rd frame!

# 3. Post-process the extracted mask frames to remove leakage!
# (The run_inference_video method saves intermediate images to the output folder)
mask_images_dir = "/home/jaskirats2004/processed_outputs/tape_video_30sec/pred_obj1"
pipeline.apply_post_processing(mask_images_dir)
```
