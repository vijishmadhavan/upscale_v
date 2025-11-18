#!/usr/bin/env python3
"""
Standalone SeedVR2 Video Upscaler CLI Script
"""

import sys
import os
import argparse
import time
import multiprocessing as mp
# Ensure safe CUDA usage with multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)
# -------------------------------------------------------------
# 1) Gestion VRAM (cudaMallocAsync) déjà en place
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# 2) Pré-parse de la ligne de commande pour récupérer --cuda_device
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda_device", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.cuda_device is not None:
    device_list_env = [x.strip() for x in _pre_args.cuda_device.split(',') if x.strip()!='']
    if len(device_list_env) == 1:
        # Single GPU: restrict visibility now
        os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

# -------------------------------------------------------------
# 3) Imports lourds (torch, etc.) après la configuration env
import torch
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from src.utils.downloads import download_weight

# Add project root to sys.path for src module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
root_dir = os.path.join(script_dir, '..', '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def apply_blur_to_frame(frame, blur_type='gaussian', blur_strength=1.0):
    """Apply various types of blur to a frame
    
    Args:
        frame: numpy array (BGR format from cv2)
        blur_type: 'gaussian', 'motion', 'box', or 'bilateral'
        blur_strength: float, intensity of blur (0-10 scale)
        
    Returns:
        Blurred frame
    """
    if blur_strength <= 0:
        return frame
        
    if blur_type == 'gaussian':
        # Gaussian blur - most natural looking
        kernel_size = max(3, int(blur_strength * 3) * 2 + 1)  # Ensure odd number
        sigma = blur_strength
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
        
    elif blur_type == 'motion':
        # Motion blur - simulates camera/object movement
        size = max(3, int(blur_strength * 5))
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        blurred = cv2.filter2D(frame, -1, kernel)
        
    elif blur_type == 'box':
        # Box blur - simple averaging
        kernel_size = max(3, int(blur_strength * 3))
        blurred = cv2.boxFilter(frame, -1, (kernel_size, kernel_size))
        
    elif blur_type == 'bilateral':
        # Bilateral filter - preserves edges while blurring
        d = max(5, int(blur_strength * 3))
        sigma_color = blur_strength * 20
        sigma_space = blur_strength * 20
        blurred = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
        
    else:
        # Default to gaussian
        return apply_blur_to_frame(frame, 'gaussian', blur_strength)
        
    return blurred

def extract_frames_from_video(video_path, debug=False, skip_first_frames=0, load_cap=None, blur_type='none', blur_strength=0.0, input_max_resolution=520):
    """
    Extract frames from video and convert to tensor format
    
    Args:
        video_path (str): Path to input video
        debug (bool): Enable debug logging
        skip_first_frame (bool): Skip the first frame during extraction
        load_cap (int): Maximum number of frames to load (None for all)
        blur_type (str): Type of blur to apply ('none', 'gaussian', 'motion', 'box', 'bilateral')
        blur_strength (float): Blur strength (0-10 scale)
        input_max_resolution (int): Maximum input resolution (longest side in pixels)
        
    Returns:
        torch.Tensor: Frames tensor in format [T, H, W, C] (Float16, normalized 0-1)
    """
    if debug:
        print(f"🎬 Extracting frames from video: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize input to specified max resolution, model will upscale to output
    if debug:
        print(f"🔧 DEBUG: input_max_resolution = {input_max_resolution}")
    if input_max_resolution > 0 and max(width, height) > input_max_resolution:
        if width > height:
            new_width = input_max_resolution
            new_height = int(height * input_max_resolution / width)
        else:
            new_height = input_max_resolution
            new_width = int(width * input_max_resolution / height)
        
        # Ensure dimensions are always > 0 and even
        new_width = max(2, new_width)
        new_height = max(2, new_height)
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1
        
        resize_input = True
        if debug:
            print(f"📏 Resizing input from {width}x{height} to {new_width}x{new_height} (max {input_max_resolution}p for model upscaling)")
    else:
        new_width, new_height = width, height
        resize_input = False
    
    if debug:
        print(f"📊 Video info: {frame_count} frames, {width}x{height}, {fps:.2f} FPS")
        if skip_first_frames:
            print(f"⏭️ Will skip first {skip_first_frames} frames")
        if load_cap:
            print(f"🔢 Will load maximum {load_cap} frames")
        if blur_type != 'none' and blur_strength > 0:
            print(f"🌀 Blur: {blur_type} (strength: {blur_strength})")
    
    frames = []
    frame_idx = 0
    frames_loaded = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip first frame if requested
        if frame_idx < skip_first_frames:
            frame_idx += 1
            if debug:
                print(f"⏭️ Skipped first frame")
            continue
        
        # Check load cap
        if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
            if debug:
                print(f"🔢 Reached load cap of {load_cap} frames")
            break
        
        # Resize frame if needed (in BGR format)
        if resize_input:
            if debug and frame_idx == skip_first_frames:  # Only print once
                print(f"🔧 Resizing frames to {new_width}x{new_height}")
            if new_width <= 0 or new_height <= 0:
                raise ValueError(f"Invalid resize dimensions: {new_width}x{new_height}")
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply blur if requested (in BGR format)
        if blur_type != 'none' and blur_strength > 0:
            frame = apply_blur_to_frame(frame, blur_type, blur_strength)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to 0-1
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        frame_idx += 1
        frames_loaded += 1
        
        if debug and frames_loaded % 100 == 0:
            total_to_load = min(frame_count, load_cap) if load_cap else frame_count
            print(f"📹 Extracted {frames_loaded}/{total_to_load} frames")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    if debug:
        print(f"✅ Extracted {len(frames)} frames")
    
    # Convert to tensor [T, H, W, C] and cast to Float16 for ComfyUI compatibility
    frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
    
    if debug:
        print(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
    
    return frames_tensor, fps


def save_frames_to_video(frames_tensor, output_path, fps=30.0, debug=False):
    """
    Save frames tensor to video file
    
    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_path (str): Output video path
        fps (float): Output video FPS
        debug (bool): Enable debug logging
    """
    if debug:
        print(f"🎬 Saving {frames_tensor.shape[0]} frames to video: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensor to numpy and denormalize
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)
    
    # Get video properties
    T, H, W, C = frames_np.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create video writer for: {output_path}")
    
    # Write frames
    for i, frame in enumerate(frames_np):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        if debug and (i + 1) % 100 == 0:
            print(f"💾 Saved {i + 1}/{T} frames")
    
    out.release()
    
    if debug:
        print(f"✅ Video saved successfully: {output_path}")


def save_frames_to_png(frames_tensor, output_dir, base_name, debug=False):
    """
    Save frames tensor as sequential PNG images.

    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_dir (str): Directory to save PNGs
        base_name (str): Base name for output files (without extension)
        debug (bool): Enable debug logging
    """
    if debug:
        print(f"🖼️ Saving {frames_tensor.shape[0]} frames as PNGs to directory: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy uint8 RGB
    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    total = frames_np.shape[0]
    digits = max(5, len(str(total)))  # at least 5 digits

    for idx, frame in enumerate(frames_np):
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = f"{base_name}_{idx:0{digits}d}.png"
        file_path = os.path.join(output_dir, filename)
        cv2.imwrite(file_path, frame_bgr)
        if debug and (idx + 1) % 100 == 0:
            print(f"💾 Saved {idx + 1}/{total} PNGs")

    if debug:
        print(f"✅ PNG saving completed: {total} files in '{output_dir}'")


def _worker_process(proc_idx, device_id, frames_np, shared_args, return_queue):
    """Worker process that performs upscaling on a slice of frames using a dedicated GPU."""
    # 1. Limit CUDA visibility to the chosen GPU BEFORE importing torch-heavy deps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Keep same cudaMallocAsync setting
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    import torch  # local import inside subprocess
    from src.core.model_manager import configure_runner
    from src.core.generation import generation_loop
    

    # Reconstruct frames tensor
    frames_tensor = torch.from_numpy(frames_np).to(torch.float16)

    # Prepare runner
    model_dir = shared_args["model_dir"]
    model_name = shared_args["model"]
    # ensure model weights present (each process checks but very fast if already downloaded)
    if shared_args["debug"]:
        print(f"🔄 Configuring runner for device {device_id}")
    runner = configure_runner(model_name, model_dir, shared_args["preserve_vram"], shared_args["debug"])

    # Run generation
    result_tensor = generation_loop(
        runner=runner,
        images=frames_tensor,
        cfg_scale=shared_args["cfg_scale"],
        seed=shared_args["seed"],
        res_w=shared_args["res_w"],
        batch_size=shared_args["batch_size"],
        preserve_vram=shared_args["preserve_vram"],
        temporal_overlap=shared_args["temporal_overlap"],
        debug=shared_args["debug"],
    )

    # Send back result as numpy array to avoid CUDA transfers
    return_queue.put((proc_idx, result_tensor.cpu().numpy()))


def _gpu_processing(frames_tensor, device_list, args):
    """Split frames and process them in parallel on multiple GPUs."""
    num_devices = len(device_list)
    # split frames tensor along time dimension
    chunks = torch.chunk(frames_tensor, num_devices, dim=0)

    manager = mp.Manager()
    return_queue = manager.Queue()
    workers = []

    shared_args = {
        "model": args.model,
        "model_dir": args.model_dir if args.model_dir is not None else "./models/SEEDVR2",
        "preserve_vram": args.preserve_vram,
        "debug": args.debug,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "res_w": args.resolution,
        "batch_size": args.batch_size,
        "temporal_overlap": 0,
    }

    for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
        p = mp.Process(
            target=_worker_process,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)

    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, res_np = return_queue.get()
        results_np[proc_idx] = res_np
        collected += 1

    for p in workers:
        p.join()

    # Concatenate results in original order
    result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float16)
    return result_tensor


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SeedVR2 Video Upscaler CLI")
    
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for generation (default: 100)")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="CFG scale for generation (default: 1.0)")
    parser.add_argument("--resolution", type=int, default=1072,
                        help="Target resolution of the short side (default: 1072)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of frames per batch (default: 5)")
    parser.add_argument("--model", type=str, default="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                        choices=[
                            "seedvr2_ema_3b_fp16.safetensors",
                            "seedvr2_ema_3b_fp8_e4m3fn.safetensors", 
                            "seedvr2_ema_7b_fp16.safetensors",
                            "seedvr2_ema_7b_fp8_e4m3fn.safetensors"
                        ],
                        help="Model to use (default: 3B FP8)")
    parser.add_argument("--model_dir", type=str, default="seedvr2_models",
                            help="Directory containing the model files (default: use cache directory)")
    parser.add_argument("--skip_first_frames", type=int, default=0,
                        help="Skip the first frames during processing")
    parser.add_argument("--load_cap", type=int, default=0,
                        help="Maximum number of frames to load from video (default: load all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated, if output_format is png, it will be a directory)")
    parser.add_argument("--output_format", type=str, default="video", choices=["video", "png"],
                        help="Output format: 'video' (mp4) or 'png' images (default: video)")
    parser.add_argument("--preserve_vram", action="store_true",
                        help="Enable VRAM preservation mode")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--cuda_device", type=str, default=None,
                        help="CUDA device id(s). Single id (e.g., '0') or comma-separated list '0,1' for multi-GPU")
    parser.add_argument("--blur_type", type=str, default="none", 
                        choices=["none", "gaussian", "motion", "box", "bilateral"],
                        help="Type of blur to apply to input (default: none)")
    parser.add_argument("--blur_strength", type=float, default=0.0,
                        help="Blur strength (0-10 scale, default: 0.0)")
    parser.add_argument("--input_max_resolution", type=int, default=520,
                        help="Maximum input resolution (longest side in pixels, default: 520)")
    
    return parser.parse_args()


def main():
    """Main CLI function"""
    print(f"🚀 SeedVR2 Video Upscaler CLI started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    args = parse_arguments()
    
    if args.debug:
        print(f"📋 Arguments:")
        for key, value in vars(args).items():
            print(f"   {key}: {value}")
    
    if args.debug:
        # Show actual CUDA device visibility
        print(f"🖥️ CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all)')}")
        if torch.cuda.is_available():
            print(f"🖥️ torch.cuda.device_count(): {torch.cuda.device_count()}")
            print(f"🖥️ Using device index 0 inside script (mapped to selected GPU)")
    
    try:
        # Ensure --output is a directory when using PNG format
        if args.output_format == "png":
            output_path_obj = Path(args.output)
            if output_path_obj.suffix:  # an extension is present, strip it
                args.output = str(output_path_obj.with_suffix(''))
        
        if args.debug:
            print(f"📁 Output will be saved to: {args.output}")
        
        # Extract frames from video
        print(f"🎬 Extracting frames from video...")
        start_time = time.time()
        frames_tensor, original_fps = extract_frames_from_video(
            args.video_path, 
            args.debug, 
            args.skip_first_frames, 
            args.load_cap,
            args.blur_type,
            args.blur_strength,
            args.input_max_resolution
        )
        
        if args.debug:
            print(f"🔄 Frame extraction time: {time.time() - start_time:.2f}s")
            # print(f"📊 Initial VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB") # may initialize cuda
        
        # Parse GPU list
        device_list = [d.strip() for d in str(args.cuda_device).split(',') if d.strip()] if args.cuda_device else ["0"]
        if args.debug:
            print(f"🚀 Using devices: {device_list}")
        processing_start = time.time()
        download_weight(args.model, args.model_dir)
        result = _gpu_processing(frames_tensor, device_list, args)
        generation_time = time.time() - processing_start
        
        if args.debug:
            print(f"🔄 Generation time: {generation_time:.2f}s")
            print(f"📊 Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            print(f"📊 Result shape: {result.shape}, dtype: {result.dtype}")
        
        # After generation_time calculation, choose saving method
        if args.output_format == "png":
            # Ensure output treated as directory
            output_dir = args.output
            base_name = Path(args.video_path).stem + "_upscaled"
            if args.debug:
                print(f"🖼️ Saving PNG frames to directory: {output_dir}")
            save_start = time.time()
            save_frames_to_png(result, output_dir, base_name, args.debug)
            if args.debug:
                print(f"🔄 Save time: {time.time() - save_start:.2f}s")
        else:
            # Save video
            if args.debug:
                print(f"💾 Saving upscaled video to: {args.output}")
            save_start = time.time()
            save_frames_to_video(result, args.output, original_fps, args.debug)
            if args.debug:
                print(f"🔄 Save time: {time.time() - save_start:.2f}s")
        
        total_time = time.time() - start_time
        print(f"✅ Upscaling completed successfully!")
        if args.output_format == "png":
            print(f"📁 PNG frames saved in directory: {args.output}")
        else:
            print(f"📁 Output saved to video: {args.output}")
        print(f"🕒 Total processing time: {total_time:.2f}s")
        print(f"⚡ Average FPS: {len(frames_tensor) / generation_time:.2f} frames/sec")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        print(f"🧹 Process {os.getpid()} terminating - VRAM will be automatically freed")


if __name__ == "__main__":
    main() 