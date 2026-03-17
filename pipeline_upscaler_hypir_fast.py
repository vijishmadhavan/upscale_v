#!/usr/bin/env python3
"""
Ultimate 3-Stage Video Enhancement Pipeline: FlashVSR + SeedVR2 + RealESRGAN (In-Memory)
Modified pipeline with FlashVSR in Stage 1:
1. FlashVSR: Fast video super-resolution with built-in temporal consistency (optional)
2. SeedVR2: Advanced AI enhancement and upscaling up to 1080p
3. RealESRGAN: Final upscaling to 4K only (1080p/1440p targets skip this stage)
Optimized for videos with smart in-memory processing for maximum speed.
"""

import sys
import os
import argparse
import time
import multiprocessing as mp
# Ensure safe CUDA usage with multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# VRAM management
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# Pre-parse critical flags before heavy imports
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda_device", type=str, default=None)
_pre_parser.add_argument("--debug", action="store_true", default=False)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.cuda_device is not None:
    device_list_env = [x.strip() for x in _pre_args.cuda_device.split(',') if x.strip()!='']
    if len(device_list_env) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

# Suppress verbose output early (before module-level prints in imports)
if not _pre_args.debug:
    os.environ["TQDM_DISABLE"] = "1"

import torch
import cv2
import numpy as np
import gc
import math
from datetime import datetime
from pathlib import Path
import subprocess
import tempfile
import shutil
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from accelerate.utils import set_seed

# Add project root to sys.path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# DBNet imports for text detection (MUST be before FlashVSR path additions to avoid import conflicts)
try:
    from nets import nn
    from utils import util
    DBNET_AVAILABLE = True
except ImportError as e:
    DBNET_AVAILABLE = False
    print(f"⚠️ DBNet import failed: {e}")

# Add FlashVSR to Python path (after DBNet imports to avoid utils conflict)
flashvsr_root = os.path.join(script_dir, 'FlashVSR')
flashvsr_examples = os.path.join(flashvsr_root, 'examples', 'WanVSR')
if flashvsr_root not in sys.path:
    sys.path.insert(0, flashvsr_root)
if flashvsr_examples not in sys.path:
    sys.path.insert(0, flashvsr_examples)

# Import new inference_cli module dynamically (all functions now from new implementation)
import importlib.util
inference_cli_new_path = os.path.join(script_dir, 'inference_cli.py')
spec = importlib.util.spec_from_file_location("inference_cli_new", inference_cli_new_path)
inference_cli_new = importlib.util.module_from_spec(spec)
sys.modules["inference_cli_new"] = inference_cli_new
spec.loader.exec_module(inference_cli_new)

# Import all needed functions from new inference_cli
apply_blur_to_frame = inference_cli_new.apply_blur_to_frame
save_frames_to_video = inference_cli_new.save_frames_to_video
save_frames_to_png = inference_cli_new.save_frames_to_png

# RealESRGAN imports (still needed for Stage 3)
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer
try:
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
except ImportError:
    SRVGGNetCompact = None

# FlashVSR imports (optional - requires Ampere GPU or newer)
try:
    from diffsynth import ModelManager, FlashVSRFullPipeline
    from utils.utils import Buffer_LQ4x_Proj
    from einops import rearrange
    FLASHVSR_AVAILABLE = True
except ImportError as e:
    FLASHVSR_AVAILABLE = False
    print(f"⚠️ FlashVSR not available: {e}")

# YOLOv8 imports for person detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


_progress = None

class ProgressBar:
    """Single unified progress bar spanning all pipeline stages."""
    def __init__(self):
        self._percent = 0
        self.ranges = {}

    def update(self, percent):
        percent = max(0, min(int(percent), 100))
        if percent == self._percent:
            return
        self._percent = percent
        filled = percent * 30 // 100
        bar = '\u2588' * filled + '\u2591' * (30 - filled)
        sys.stdout.write(f'\r\u23f3 Upscaling [{bar}] {percent}%')
        sys.stdout.flush()

    def stage(self, name, fraction):
        """Update progress within a named stage. fraction: 0.0 to 1.0"""
        if name in self.ranges:
            s, e = self.ranges[name]
            self.update(s + (e - s) * max(0.0, min(1.0, fraction)))

    def complete(self):
        self.update(100)
        sys.stdout.write('\n')
        sys.stdout.flush()


def open_video_writer(path, width, height, fps, debug=False):
    """Create a cv2.VideoWriter with codec fallback, suppressing stderr noise from failed codecs."""
    writer = None
    for codec in ['h264', 'avc1', 'H264', 'X264', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Suppress C-level stderr from OpenCV/FFMPEG codec probing
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
        if writer.isOpened():
            if debug:
                print(f"📹 Using codec: {codec}")
            return writer
    raise RuntimeError(f"Failed to open video writer for {path}")


def detect_persons_in_video(video_path, confidence_threshold=0.5, max_frames_to_check=30, debug=False, model_dir=None):
    """
    Detect if there are people in the video using YOLOv8

    Args:
        video_path (str): Path to input video
        confidence_threshold (float): Minimum confidence for person detection
        max_frames_to_check (int): Maximum number of frames to analyze
        debug (bool): Enable debug logging

    Returns:
        bool: True if people are detected, False otherwise
    """
    if not YOLO_AVAILABLE:
        return True  # Default to processing if YOLO not available

    try:
        # Load YOLOv8 model
        model = YOLO("yolov8n.pt")  # Downloads automatically if not present

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return True  # Default to processing if video can't be opened

        # Get total frames and calculate sampling interval
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames_to_check)

        person_detected = False
        frames_checked = 0

        frame_idx = 0
        while frames_checked < max_frames_to_check:
            # Skip to next sampling frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Run YOLO inference
            results = model(frame, verbose=False)

            # Check for person detections (class 0 in COCO dataset)
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls == 0 and conf > confidence_threshold:  # Person class
                            person_detected = True
                            break

                if person_detected:
                    break

            if person_detected:
                break

            frames_checked += 1
            frame_idx += frame_interval

        cap.release()

        if debug:
            print(f"👤 Person detection: {'People detected' if person_detected else 'No people detected'}")

        return person_detected

    except Exception as e:
        if debug:
            print(f"⚠️ Person detection failed: {e}")
        return True  # Default to processing if error occurs


def detect_text_in_video(video_path, max_frames_to_check=None, debug=False, model_dir=None, script_dir=None):
    """
    Detect text regions in video frames using DBNet

    Args:
        video_path (str): Path to input video
        max_frames_to_check (int): Maximum number of frames to analyze. None = check all frames
        debug (bool): Enable debug logging
        model_dir (str): Directory containing models

    Returns:
        tuple: (text_regions dict, avg_coverage_percent)
            - text_regions: {frame_idx: [list of text polygons], ...} for frames with text detected
            - avg_coverage_percent: Average percentage of frame covered by text
    """
    if not DBNET_AVAILABLE:
        if debug:
            print("⚠️ DBNet not available - skipping text detection")
        return {}, 0.0

    try:
        # Use script directory if provided, otherwise fall back to current directory
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load DBNet model
        model_candidates = [
            os.path.join(model_dir, 'weights', 'last.pt') if model_dir else None,
            os.path.join(script_dir, 'weights', 'last.pt'),
            './weights/last.pt',
        ]

        # Filter out None values
        model_candidates = [c for c in model_candidates if c is not None]

        if debug:
            print(f"🔍 Searching for DBNet model in:")
            for c in model_candidates:
                exists = "✓" if os.path.exists(c) else "✗"
                print(f"  {exists} {c}")

        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

        if not model_path:
            if debug:
                print("⚠️ DBNet model not found - skipping text detection")
            return {}, 0.0

        if debug:
            print(f"✅ Using DBNet model: {model_path}")

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model = model.float()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # Normalization parameters (same as video_demo.py)
        mean = np.array([0.406, 0.456, 0.485]).reshape((1, 1, 3)).astype('float32')
        std = np.array([0.225, 0.224, 0.229]).reshape((1, 1, 3)).astype('float32')
        threshold = 0.3

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}, 0.0

        try:
            # Get total frames and frame dimensions
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_area = frame_width * frame_height

            # Use smaller input size for long videos (faster detection)
            input_size = 384 if total_frames > 800 else 800
            if debug and total_frames > 800:
                print(f"📊 Long video ({total_frames} frames) - using input_size={input_size} for faster text detection")

            # If max_frames_to_check is None, check all frames
            if max_frames_to_check is None:
                max_frames_to_check = total_frames
                frame_interval = 1
            else:
                frame_interval = max(1, total_frames // max_frames_to_check)

            if debug:
                print(f"📊 Checking {min(max_frames_to_check, total_frames)} of {total_frames} frames for text")

            text_regions = {}
            frames_checked = 0
            total_coverage = 0.0
            frames_with_text = 0

            frame_idx = 0
            while frames_checked < max_frames_to_check and frame_idx < total_frames:
                # Skip to next sampling frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # Prepare image for DBNet model (same as video_demo.py)
                shape = frame.shape[:2]
                width = int(shape[1] * input_size / shape[0])
                width = math.ceil(width / 32) * 32

                x = cv2.resize(frame, dsize=(width, input_size))
                x = x.astype('float32') / 255.0
                x = x - mean
                x = x / std
                x = x.transpose((2, 0, 1))[::-1]
                x = np.ascontiguousarray(x)
                x = torch.from_numpy(x).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    output = model(x)
                    output = util.mask_to_box(
                        targets={'shape': [shape]},
                        outputs=output.cpu(),
                        threshold=threshold,
                        is_polygon=True
                    )
                    boxes, scores = output[0][0], output[1][0]

                if len(boxes) > 0:
                    # Convert boxes to polygon format (list of lists of [x, y] coordinates)
                    polygons = []
                    for box in boxes:
                        poly = np.array(box).reshape((-1, 2)).tolist()
                        polygons.append(poly)

                    # Store polygons for this frame
                    text_regions[frame_idx] = polygons

                    # Calculate text coverage percentage for this frame
                    text_area = 0
                    for poly in polygons:
                        poly_array = np.array(poly, dtype=np.int32)
                        x_min = max(0, int(np.min(poly_array[:, 0])))
                        y_min = max(0, int(np.min(poly_array[:, 1])))
                        x_max = min(frame_width, int(np.max(poly_array[:, 0])))
                        y_max = min(frame_height, int(np.max(poly_array[:, 1])))
                        text_area += (x_max - x_min) * (y_max - y_min)

                    coverage_percent = (text_area / frame_area) * 100
                    total_coverage += coverage_percent
                    frames_with_text += 1

                frames_checked += 1
                frame_idx += frame_interval
        finally:
            cap.release()

        # Clean up model
        del model
        torch.cuda.empty_cache()

        # Calculate average coverage
        avg_coverage = total_coverage / frames_with_text if frames_with_text > 0 else 0.0

        return text_regions, avg_coverage

    except Exception as e:
        print(f"⚠️ Text detection failed: {e}")
        return {}, 0.0


def extract_audio(video_path, audio_path, debug=False):
    """
    Extract audio from video file using ffmpeg
    
    Args:
        video_path (str): Path to input video
        audio_path (str): Path to save extracted audio
        debug (bool): Enable debug logging
        
    Returns:
        bool: True if audio was extracted, False if no audio track
    """
    try:
        # Check if video has audio stream
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type', '-of', 'default=nw=1:nk=1',
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)

        if result.stdout.strip() != 'audio':
            return False

        # Extract audio
        cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'copy',
            '-y', audio_path
        ]

        if not debug:
            cmd.extend(['-loglevel', 'error'])

        subprocess.run(cmd, check=True)
        return True

    except subprocess.CalledProcessError as e:
        return False
    except FileNotFoundError:
        return False


def merge_audio_video(video_path, audio_path, output_path, debug=False):
    """
    Merge audio track with video using ffmpeg

    Args:
        video_path (str): Path to video file (no audio)
        audio_path (str): Path to audio file
        output_path (str): Path for output video with audio
        debug (bool): Enable debug logging
    """
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'copy', '-shortest',
            '-y', output_path
        ]

        if not debug:
            cmd.extend(['-loglevel', 'error'])

        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        raise


def save_frames_to_video_with_audio(frames_tensor, output_path, fps=30.0, debug=False, audio_path=None):
    """
    Save frames tensor to video file with optional audio
    
    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_path (str): Output video path
        fps (float): Output video FPS
        debug (bool): Enable debug logging
        audio_path (str): Optional path to audio file to merge
    """
    # First save video without audio
    if audio_path and os.path.exists(audio_path):
        # Save to temp file first
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        save_frames_to_video(frames_tensor, temp_video, fps, debug)
        
        # Merge with audio
        try:
            merge_audio_video(temp_video, audio_path, output_path, debug)
            os.unlink(temp_video)  # Clean up temp file
        except Exception as e:
            # If merge fails, just copy video without audio
            shutil.move(temp_video, output_path)
            if debug:
                print(f"⚠️ Audio merge failed, saved video without audio: {e}")
    else:
        # No audio to merge, save directly
        save_frames_to_video(frames_tensor, output_path, fps, debug)


def save_tensor_to_video_streaming(frames_tensor, output_path, fps, debug=False):
    """
    Production-ready streaming save for large tensors
    Writes frames one-by-one to avoid memory accumulation

    Args:
        frames_tensor: Tensor [T, H, W, C] in float16/32, range [0,1]
        output_path: Output video file path
        fps: Frame rate
        debug: Debug logging
    """
    if debug:
        print(f"🎬 Streaming save to video: {output_path}")
        print(f"📊 Tensor shape: {frames_tensor.shape}")

    T, H, W, C = frames_tensor.shape

    # Create video writer with codec fallback (stderr suppressed)
    writer = open_video_writer(output_path, W, H, fps, debug)

    try:
        # Process frames one at a time to avoid memory spike
        for i in range(T):
            # Get single frame and convert to uint8
            frame = frames_tensor[i].cpu().numpy()
            frame_uint8 = (frame * 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

            # Write frame
            writer.write(frame_bgr)

            # Periodic progress update
            if debug and (i + 1) % 100 == 0:
                print(f"📝 Written {i + 1}/{T} frames")

            # Aggressive memory cleanup every 500 frames
            if (i + 1) % 500 == 0:
                gc.collect()

    finally:
        writer.release()

    if debug:
        print(f"✅ Streaming save complete: {output_path}")


def process_seedvr2_adaptive(frames_tensor, args, original_fps, device_list, seedvr2_args, debug=False, threshold=None):
    """
    Production-ready SeedVR2 processing with adaptive output strategy

    Returns:
        tuple: (tensor_result, temp_file_path)
            - If short video: (tensor, None)
            - If long video: (None, temp_file_path)
    """
    frame_count = frames_tensor.shape[0]

    # Use provided threshold or default
    max_threshold = threshold if threshold is not None else getattr(args, 'streaming_threshold', 800)

    # Determine safe frame limit based on GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # Conservative: ~50 frames per GB for 1280p processing
        safe_frame_limit = int(gpu_memory_gb * 50)
        safe_frame_limit = max(200, min(safe_frame_limit, max_threshold))  # Clamp between 200 and threshold
    else:
        safe_frame_limit = 200

    if debug:
        print(f"📊 Frame count: {frame_count}, Safe limit: {safe_frame_limit}")

    if frame_count <= safe_frame_limit:
        # FAST PATH: Process all frames in memory for short videos
        if debug:
            print(f"✅ Short video mode: Processing {frame_count} frames in memory")

        result = inference_cli_new._gpu_processing(frames_tensor, device_list, seedvr2_args)
        return result, None

    else:
        # SAFE PATH: Process in batches and stream to disk for long videos
        if debug:
            print(f"💾 Long video mode: Processing {frame_count} frames with disk streaming")

        # Create temp file for output
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        # Get output dimensions after SeedVR2 processing
        # Process first frame to get output size
        test_batch = frames_tensor[:1]
        test_output = inference_cli_new._gpu_processing(test_batch, device_list, seedvr2_args)
        _, out_h, out_w, _ = test_output.shape
        del test_output
        torch.cuda.empty_cache()

        # Create video writer with codec fallback (stderr suppressed)
        writer = open_video_writer(temp_file, out_w, out_h, original_fps, debug)

        # Process in optimal batches
        batch_size = min(seedvr2_args.batch_size, safe_frame_limit // 2)
        batch_size = max(5, batch_size)  # At least 5 frames per batch

        for i in range(0, frame_count, batch_size):
            batch_end = min(i + batch_size, frame_count)
            batch = frames_tensor[i:batch_end]

            # Process batch through SeedVR2
            processed_batch = inference_cli_new._gpu_processing(batch, device_list, seedvr2_args)

            # Write immediately to disk
            for j in range(processed_batch.shape[0]):
                frame = processed_batch[j].cpu().numpy()
                frame_uint8 = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            del processed_batch
            torch.cuda.empty_cache()
            gc.collect()

            if _progress:
                _progress.stage('seedvr2', batch_end / frame_count)
            if debug:
                print(f"✓ Processed {batch_end}/{frame_count} frames")

        writer.release()
        return None, temp_file


def clear_gpu_memory(preserve_data=False):
    """Smart GPU memory management"""
    if torch.cuda.is_available():
        if preserve_data:
            # Light cleanup - just clear unused allocations
            torch.cuda.empty_cache()
            gc.collect()
        else:
            # Full cleanup - clear everything
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            for _ in range(3):
                gc.collect()
            torch.cuda.empty_cache()


def get_gpu_memory_info():
    """Get current GPU memory usage info"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def create_seedvr2_args_from_pipeline(pipeline_args, model_dir=None):
    """
    Create argparse.Namespace with new SeedVR2 CLI arguments from pipeline args

    Args:
        pipeline_args: Pipeline argument namespace
        model_dir: Optional model directory override

    Returns:
        argparse.Namespace configured for new inference_cli processing
    """
    # Get model from registry
    from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE
    from src.utils.constants import SEEDVR2_FOLDER_NAME

    # Create new args namespace with all required parameters
    seedvr2_args = argparse.Namespace()

    # Model configuration
    seedvr2_args.dit_model = getattr(pipeline_args, 'model', DEFAULT_DIT)
    seedvr2_args.model_dir = model_dir or (
        pipeline_args.model_dir if hasattr(pipeline_args, 'model_dir') and pipeline_args.model_dir
        else f"./models/{SEEDVR2_FOLDER_NAME}"
    )

    # Processing parameters
    seedvr2_args.resolution = getattr(pipeline_args, 'resolution', 1280)  # Short-side target
    seedvr2_args.max_resolution = getattr(pipeline_args, 'max_resolution', 1280)  # Cap at 1280p
    seedvr2_args.batch_size = getattr(pipeline_args, 'batch_size', 5)
    seedvr2_args.uniform_batch_size = True  # Enable for best quality
    seedvr2_args.seed = getattr(pipeline_args, 'seed', 42)
    seedvr2_args.skip_first_frames = getattr(pipeline_args, 'skip_first_frames', 0)
    seedvr2_args.load_cap = getattr(pipeline_args, 'load_cap', 0)
    seedvr2_args.prepend_frames = getattr(pipeline_args, 'prepend_frames', 0)
    seedvr2_args.temporal_overlap = getattr(pipeline_args, 'temporal_overlap', 0)

    # Quality control
    seedvr2_args.color_correction = getattr(pipeline_args, 'color_correction', 'lab')
    seedvr2_args.input_noise_scale = getattr(pipeline_args, 'input_noise_scale', 0.0)
    seedvr2_args.latent_noise_scale = getattr(pipeline_args, 'latent_noise_scale', 0.0)

    # Device management
    seedvr2_args.cuda_device = getattr(pipeline_args, 'cuda_device', None)
    seedvr2_args.dit_offload_device = getattr(pipeline_args, 'dit_offload_device', 'none')
    seedvr2_args.vae_offload_device = getattr(pipeline_args, 'vae_offload_device', 'none')
    seedvr2_args.tensor_offload_device = getattr(pipeline_args, 'tensor_offload_device', 'cpu')

    # Memory optimization (BlockSwap)
    seedvr2_args.blocks_to_swap = getattr(pipeline_args, 'blocks_to_swap', 0)
    seedvr2_args.swap_io_components = getattr(pipeline_args, 'swap_io_components', False)

    # VAE tiling
    seedvr2_args.vae_encode_tiled = getattr(pipeline_args, 'vae_encode_tiled', False)
    seedvr2_args.vae_encode_tile_size = getattr(pipeline_args, 'vae_encode_tile_size', 1024)
    seedvr2_args.vae_encode_tile_overlap = getattr(pipeline_args, 'vae_encode_tile_overlap', 128)
    seedvr2_args.vae_decode_tiled = getattr(pipeline_args, 'vae_decode_tiled', False)
    seedvr2_args.vae_decode_tile_size = getattr(pipeline_args, 'vae_decode_tile_size', 1024)
    seedvr2_args.vae_decode_tile_overlap = getattr(pipeline_args, 'vae_decode_tile_overlap', 128)
    seedvr2_args.tile_debug = getattr(pipeline_args, 'tile_debug', 'false')

    # Performance optimization
    seedvr2_args.attention_mode = getattr(pipeline_args, 'attention_mode', 'sdpa')
    seedvr2_args.compile_dit = getattr(pipeline_args, 'compile_dit', False)
    seedvr2_args.compile_vae = getattr(pipeline_args, 'compile_vae', False)
    seedvr2_args.compile_backend = getattr(pipeline_args, 'compile_backend', 'inductor')
    seedvr2_args.compile_mode = getattr(pipeline_args, 'compile_mode', 'default')
    seedvr2_args.compile_fullgraph = getattr(pipeline_args, 'compile_fullgraph', False)
    seedvr2_args.compile_dynamic = getattr(pipeline_args, 'compile_dynamic', False)
    seedvr2_args.compile_dynamo_cache_size_limit = getattr(pipeline_args, 'compile_dynamo_cache_size_limit', 64)
    seedvr2_args.compile_dynamo_recompile_limit = getattr(pipeline_args, 'compile_dynamo_recompile_limit', 128)

    # Model caching (for batch processing)
    seedvr2_args.cache_dit = getattr(pipeline_args, 'cache_dit', False)
    seedvr2_args.cache_vae = getattr(pipeline_args, 'cache_vae', False)

    # Debugging
    seedvr2_args.debug = getattr(pipeline_args, 'debug', False)
    seedvr2_args.quiet = not getattr(pipeline_args, 'debug', False)

    return seedvr2_args


def resize_input_frames_to_720p(frames_tensor, debug=False):
    """Resize input frames so both dimensions are ≤620p while maintaining aspect ratio"""
    T, H, W, C = frames_tensor.shape
    max_dimension = 620

    # Calculate scale factor based on longest side
    longest_side = max(H, W)

    if longest_side <= max_dimension:
        # Already within limits
        if debug:
            print(f"📏 Input already ≤620p ({W}x{H}), skipping resize")
        return frames_tensor

    # Scale down so longest side = 620p
    scale = max_dimension / longest_side
    target_width = int(W * scale)
    target_height = int(H * scale)

    # Ensure even dimensions for video encoding
    target_width = target_width if target_width % 2 == 0 else target_width + 1
    target_height = target_height if target_height % 2 == 0 else target_height + 1

    if debug:
        print(f"📏 Resizing input from {W}x{H} to {target_width}x{target_height} (max 620p)")
    
    # Convert to numpy for processing
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)
    
    # Resize frames
    resized_frames = []
    for i, frame in enumerate(frames_np):
        # Frame is in RGB format
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        resized_frames.append(resized)
        
        if debug and (i + 1) % 50 == 0:
            print(f"🔄 Resized {i + 1}/{len(frames_np)} frames")
    
    # Convert back to tensor
    result_np = np.stack(resized_frames)
    result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0).to(torch.float16)
    
    if debug:
        print(f"✅ Input resize complete: {result_tensor.shape}")

    return result_tensor


def extract_and_resize_frames(video_path, max_dimension=620, skip_first_frames=0, load_cap=None, debug=False):
    """Extract frames from video and resize in a single pass — never stores full resolution.

    Replaces extract_frames_from_video() + resize_input_frames_to_720p() with one loop.
    Uses same INTER_LANCZOS4 interpolation. Peak RAM: ~4 GB instead of ~38 GB for 1080p input.

    Returns:
        Tuple of (frames_tensor [T,H,W,C] float16 range [0,1], fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate target size — same logic as resize_input_frames_to_720p
    longest_side = max(orig_height, orig_width)
    if longest_side > max_dimension:
        scale_factor = max_dimension / longest_side
        target_width = int(orig_width * scale_factor)
        target_height = int(orig_height * scale_factor)
        target_width = target_width if target_width % 2 == 0 else target_width + 1
        target_height = target_height if target_height % 2 == 0 else target_height + 1
        needs_resize = True
    else:
        target_width = orig_width
        target_height = orig_height
        needs_resize = False

    if debug:
        if needs_resize:
            print(f"📏 Extract+resize: {orig_width}x{orig_height} → {target_width}x{target_height} (max {max_dimension}p)")
        else:
            print(f"📏 Input already ≤{max_dimension}p ({orig_width}x{orig_height}), no resize needed")

    frames = []
    frame_idx = 0
    frames_loaded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < skip_first_frames:
                frame_idx += 1
                continue

            if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
                break

            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize immediately — only small frames accumulate in RAM
            if needs_resize:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

            frames.append(frame)
            frame_idx += 1
            frames_loaded += 1

            if debug and frames_loaded % 200 == 0:
                print(f"🔄 Extracted {frames_loaded}/{frame_count} frames")
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    # Single np.stack on small uint8 frames → float16 tensor
    frames_tensor = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0).to(torch.float16)

    if debug:
        print(f"✅ Extracted {frames_loaded} frames: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")

    return frames_tensor, fps


def process_with_flashvsr(frames_tensor, args, debug=False, text_regions=None):
    """Process frames with FlashVSR following reference implementation exactly

    Args:
        frames_tensor: Input frames tensor [T, H, W, C] - already resized to 620p by pipeline
        args: Arguments
        debug: Debug flag
        text_regions: Dict of {frame_idx: [polygons]} for text protection (kept for compatibility)

    Returns:
        result_tensor: Enhanced frames tensor [T, H, W, C]
        loading_time: Time spent loading models
        inference_time: Time spent on inference
    """
    if debug:
        print(f"\n🎯 Starting FlashVSR video super-resolution")
        print(f"📊 Input tensor shape: {frames_tensor.shape}")

    loading_start = time.time()
    T, H, W, C = frames_tensor.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine scale based on resolution and frame count
    input_p = min(H, W)
    flashvsr_scale_threshold = getattr(args, 'flashvsr_scale_threshold', 500)

    if args.flashvsr_auto_mode:
        # For long videos (>500 frames), use 1x scale (enhancement only) to save time/memory
        if T > flashvsr_scale_threshold:
            scale = 1.0  # Enhancement only for long videos
            if debug:
                print(f"📊 Long video ({T} frames > {flashvsr_scale_threshold}) - Using 1x scale (enhancement only)")
        elif input_p >= 1024:
            scale = 1.0  # Enhancement only for high-res input
            if debug:
                print(f"📊 Input resolution {W}x{H} >= 1024p - Using 1x scale (enhancement only)")
        else:
            scale = 2.0  # 2x upscaling for short videos with low-res input
            if debug:
                print(f"📊 Short video ({T} frames), {W}x{H} < 1024p - Using 2x upscaling")
    else:
        scale = args.flashvsr_scale

    if debug:
        print(f"🔧 Setting up FlashVSR pipeline")

    # Initialize pipeline (same as init_pipeline() in reference)
    # Suppress DiffSynth model loading messages
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        flashvsr_models_dir = os.path.join(script_dir, 'FlashVSR', 'examples', 'WanVSR', 'FlashVSR')
        mm.load_models([
            os.path.join(flashvsr_models_dir, "diffusion_pytorch_model_streaming_dmd.safetensors"),
            os.path.join(flashvsr_models_dir, "Wan2.1_VAE.pth"),
        ])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
    finally:
        sys.stdout = old_stdout
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    LQ_proj_in_path = os.path.join(flashvsr_models_dir, "LQ_proj_in.ckpt")
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to('cuda')
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None
    pipe.to('cuda')
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])

    loading_time = time.time() - loading_start
    if debug:
        print(f"✅ FlashVSR pipeline loaded ({loading_time:.2f}s)")

    # Prepare input tensor (same as prepare_input_tensor() in reference)
    inference_start = time.time()

    # Calculate dimensions (must be multiple of 128)
    sW, sH = int(W * scale), int(H * scale)
    tW = max(128, (sW // 128) * 128)
    tH = max(128, (sH // 128) * 128)

    if debug:
        print(f"📊 Target resolution: {sW}x{sH} -> {tW}x{tH} (128-multiple)")

    # Convert to numpy
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)

    # Add 4 padding frames at the end (same as reference)
    frames_with_padding = np.concatenate([frames_np, np.repeat(frames_np[-1:], 4, axis=0)], axis=0)

    # Calculate F (same as largest_8n1_leq logic in reference)
    def largest_8n1_leq(n):
        return 0 if n < 1 else ((n - 1)//8)*8 + 1

    F = largest_8n1_leq(len(frames_with_padding))
    if F == 0:
        raise RuntimeError(f"Not enough frames. Got {len(frames_with_padding)}")
    frames_with_padding = frames_with_padding[:F]

    if debug:
        print(f"📊 Frames: {T} -> {F} (with padding for 8n+1 requirement)")

    # Upscale and crop frames (same as upscale_then_center_crop in reference)
    frames_list = []
    for frame in frames_with_padding:
        img = Image.fromarray(frame)
        # Bicubic upscale
        img_up = img.resize((sW, sH), Image.BICUBIC)
        # Center crop to 128-multiple
        l = max(0, (sW - tW) // 2)
        t = max(0, (sH - tH) // 2)
        img_crop = img_up.crop((l, t, l + tW, t + tH))
        # Convert to tensor in [-1, 1] range (pil_to_tensor_neg1_1 in reference)
        t_tensor = torch.from_numpy(np.asarray(img_crop, np.uint8)).to(device=device, dtype=torch.float32)
        t_tensor = t_tensor.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        t_tensor = t_tensor.to(torch.bfloat16)
        frames_list.append(t_tensor)

    LQ_video = torch.stack(frames_list, 0).permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F, H, W]

    # Run pipeline (same as reference main())
    sparse_ratio = getattr(args, 'flashvsr_sparse_ratio', 2.0)
    local_range = getattr(args, 'flashvsr_local_range', 11)
    tiled = getattr(args, 'flashvsr_tiled', True)  # Default to True to avoid OOM

    if debug:
        print(f"🎬 Processing with FlashVSR (sparse_ratio={sparse_ratio}, local_range={local_range}, tiled={tiled})")

    video = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
        tiled=tiled,
        LQ_video=LQ_video, num_frames=F, height=tH, width=tW, is_full_block=False, if_buffer=True,
        topk_ratio=sparse_ratio * 768 * 1280 / (tH * tW),
        kv_ratio=3.0,
        local_range=local_range,
        color_fix=True,
    )

    # Clean up input tensors immediately
    del LQ_video
    torch.cuda.empty_cache()

    # Convert output (tensor2video in reference) - do this step-by-step to manage memory
    video = rearrange(video, "C T H W -> T H W C")

    # Move to CPU first to free GPU memory before conversion
    video_cpu = video.cpu()
    del video
    torch.cuda.empty_cache()

    # Convert to numpy
    video = ((video_cpu.float() + 1) * 127.5).clip(0, 255).numpy().astype(np.uint8)
    del video_cpu

    # Remove padding frames to get back to original count
    video = video[:T]  # Remove the 4 padding frames we added

    result_tensor = torch.from_numpy(video.astype(np.float32) / 255.0).to(torch.float16)

    inference_time = time.time() - inference_start

    # Clean up
    del pipe, mm
    torch.cuda.empty_cache()

    if debug:
        print(f"✅ FlashVSR complete ({inference_time:.2f}s)")
        print(f"📊 Output shape: {result_tensor.shape}")

    return result_tensor, loading_time, inference_time


def apply_blur_and_resize_to_frames(frames_tensor, blur_type, blur_strength, max_resolution, debug=False):
    """Apply blur and resize to frames tensor for SeedVR2 input"""
    if debug:
        print(f"\n🔄 Preparing frames for SeedVR2...")
        print(f"📊 Input shape: {frames_tensor.shape}")

    # Convert to numpy
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)

    T, H, W, C = frames_tensor.shape

    # Calculate resize if needed
    if max_resolution > 0 and max(W, H) > max_resolution:
        if W > H:
            new_width = max_resolution
            new_height = int(H * max_resolution / W)
        else:
            new_height = max_resolution
            new_width = int(W * max_resolution / H)

        # Ensure dimensions are valid and even
        new_width = max(2, new_width)
        new_height = max(2, new_height)
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        if debug:
            print(f"📏 Resizing from {W}x{H} to {new_width}x{new_height} (max {max_resolution}p)")
        resize_needed = True
    else:
        new_width, new_height = W, H
        resize_needed = False

    # Process frames
    processed_frames = []
    for i, frame in enumerate(frames_np):
        # Frame is in RGB format from tensor

        # Apply resize if needed
        if resize_needed:
            if new_width <= 0 or new_height <= 0:
                raise ValueError(f"Invalid resize dimensions in apply_blur_and_resize_to_frames: {new_width}x{new_height}")
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Apply blur if requested
        if blur_type != 'none' and blur_strength > 0:
            # Convert RGB to BGR for OpenCV blur functions
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = apply_blur_to_frame(frame_bgr, blur_type, blur_strength)
            # Convert back to RGB
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        processed_frames.append(frame)

    # Convert back to tensor
    result_np = np.stack(processed_frames)
    result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0).to(torch.float16)

    if debug:
        if blur_type != 'none' and blur_strength > 0:
            print(f"🌀 Applied {blur_type} blur (strength: {blur_strength})")
        print(f"📊 Output shape: {result_tensor.shape}")

    return result_tensor



def create_streaming_video_writer(output_path, width, height, fps, debug=False, with_audio=False):
    """Create a streaming video writer using FFmpeg for maximum speed"""
    import cv2
    import subprocess

    # If audio will be added later, write to temp file
    if with_audio:
        temp_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        actual_path = temp_path
    else:
        actual_path = output_path

    # Try FFmpeg subprocess first (fastest and most reliable)
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            actual_path
        ]

        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        class FFmpegWriter:
            def __init__(self, process, temp_path=None, final_path=None):
                self.process = process
                self.temp_path = temp_path
                self.final_path = final_path

            def write(self, frame):
                try:
                    self.process.stdin.write(frame.tobytes())
                    return True
                except (OSError, BrokenPipeError) as e:
                    print(f"⚠️ FFmpeg write failed: {e}")
                    return False

            def release(self):
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.wait()

            def isOpened(self):
                return self.process.poll() is None

        if debug:
            print(f"✅ Using FFmpeg pipe writer (fast H.264): {width}x{height} @ {fps}fps")

        return FFmpegWriter(
            ffmpeg_process,
            temp_path=actual_path if with_audio else None,
            final_path=output_path if with_audio else None
        )

    except Exception as e:
        if debug:
            print(f"⚠️ FFmpeg pipe failed: {e}, using OpenCV fallback")

    # Fallback to OpenCV (slower, stderr suppressed during codec probing)
    writer = open_video_writer(actual_path, width, height, fps, debug)

    class VideoWriterWrapper:
        def __init__(self, writer, temp_path=None, final_path=None):
            self.writer = writer
            self.temp_path = temp_path
            self.final_path = final_path

        def write(self, frame):
            return self.writer.write(frame)

        def release(self):
            return self.writer.release()

        def isOpened(self):
            return self.writer.isOpened()

    return VideoWriterWrapper(
        writer,
        temp_path=actual_path if with_audio else None,
        final_path=output_path if with_audio else None
    )


def _setup_realesrgan(model_dir, target_scale, tile=0, explicit_model=None, debug=False):
    """Create RealESRGAN upsampler using native-scale RRDBNet model.

    Returns:
        (upsampler, model_name)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type == 'cuda'

    # If user specified a model explicitly, use it
    if explicit_model:
        if 'x4v3' in os.path.basename(explicit_model) and SRVGGNetCompact is not None:
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            native_scale = 4
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=target_scale)
            native_scale = target_scale
        upsampler = RealESRGANer(
            scale=native_scale, model_path=explicit_model, model=model,
            tile=tile, tile_pad=32, pre_pad=0, half=half
        )
        return upsampler, os.path.basename(explicit_model)

    # Use native-scale RRDB model (RealESRGAN_x2plus or x4plus)
    rrdb_name = f'RealESRGAN_x{target_scale}plus.pth'
    model_path = os.path.join(model_dir, rrdb_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find {rrdb_name} in {model_dir}"
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=target_scale)
    upsampler = RealESRGANer(
        scale=target_scale, model_path=model_path, model=model,
        tile=tile, tile_pad=32, pre_pad=0, half=half
    )
    if debug:
        print(f"📦 Using RealESRGAN x{target_scale}plus (RRDBNet, native {target_scale}x)")
    return upsampler, f'RealESRGAN_x{target_scale}plus'


def _get_realesrgan_batch_size(height, width):
    """Calculate safe batch size for batched RealESRGAN (RRDBNet, ~16.7M params)."""
    pixels = height * width
    if pixels <= 300_000:       # up to ~640x480
        return 4
    elif pixels <= 1_000_000:   # up to ~1280x720
        return 2
    else:                       # 1080p+
        return 1


def _batch_enhance(upsampler, frames_bgr, outscale):
    """Batch-process multiple frames through RealESRGAN model in one GPU forward pass.

    Args:
        upsampler: RealESRGANer instance (provides .model, .device, .scale, .half)
        frames_bgr: list of N frames, each (H, W, 3) uint8 BGR
        outscale: desired output scale factor

    Returns:
        list of N frames, each (H*outscale, W*outscale, 3) uint8 BGR
    """
    import torch.nn.functional as F

    n = len(frames_bgr)
    h, w = frames_bgr[0].shape[:2]
    model_scale = upsampler.scale

    # Pre-process: BGR→RGB, normalize, transpose to (3,H,W), stack to (N,3,H,W)
    tensors = []
    for frame in frames_bgr:
        img = frame[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)
        tensors.append(torch.from_numpy(np.ascontiguousarray(img)))

    batch = torch.stack(tensors).to(upsampler.device)  # (N,3,H,W)
    if upsampler.half:
        batch = batch.half()

    # Pad to make H,W divisible by model scale
    mod = model_scale
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h > 0 or pad_w > 0:
        batch = F.pad(batch, (0, pad_w, 0, pad_h), mode='reflect')

    # Forward pass — single batched inference
    with torch.no_grad():
        output = upsampler.model(batch)

    # Remove padding from output (scaled up)
    out_h = h * model_scale
    out_w = w * model_scale
    output = output[:, :, :out_h, :out_w]

    # Post-process: clamp, to numpy, transpose, RGB→BGR, uint8
    output = output.clamp_(0, 1).float().cpu().numpy()  # (N,3,H_out,W_out)

    results = []
    for i in range(n):
        img = np.transpose(output[i], (1, 2, 0))  # (H_out,W_out,3) RGB
        img_bgr = img[:, :, ::-1]  # RGB→BGR
        img_uint8 = (np.ascontiguousarray(img_bgr) * 255.0).round().astype(np.uint8)

        # Resize if outscale differs from model's native scale
        if outscale != model_scale:
            target_w = int(w * outscale)
            target_h = int(h * outscale)
            img_uint8 = cv2.resize(img_uint8, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        results.append(img_uint8)

    return results


def process_with_realesrgan_streaming(frames_input, args, output_path, fps, debug=False, audio_path=None, lanczos_2x=False):
    """Process frames with RealESRGAN using streaming output - production level.

    Args:
        frames_input: torch.Tensor [T,H,W,C] float16/float32 range [0,1] OR file path string
        lanczos_2x: If True, apply Lanczos 2x resize after each frame (folds into write loop)
    """
    if debug:
        print(f"\n🎯 Starting RealESRGAN streaming processing")

    # Track timing
    loading_start = time.time()
    inference_time = 0

    # Handle both tensor and file path input
    video_cap = None
    frames_np = None
    if isinstance(frames_input, str):
        video_cap = cv2.VideoCapture(frames_input)
        if not video_cap.isOpened():
            raise ValueError(f"Cannot open video for RealESRGAN: {frames_input}")
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if debug:
            print(f"📁 Reading frames from file: {total_frames} frames, {input_width}x{input_height}")
    else:
        if debug:
            print(f"📊 Input tensor shape: {frames_input.shape}")
        frames_np = frames_input.cpu().numpy()
        frames_np = (frames_np * 255.0).astype(np.uint8)
        total_frames = len(frames_np)
        input_height, input_width = frames_np[0].shape[:2]

    esrgan_width = input_width * args.realesrgan_scale
    esrgan_height = input_height * args.realesrgan_scale
    # Final output: double again if lanczos_2x
    output_width = esrgan_width * 2 if lanczos_2x else esrgan_width
    output_height = esrgan_height * 2 if lanczos_2x else esrgan_height

    # Setup RealESRGAN
    upsampler, model_name = _setup_realesrgan(
        args.model_dir, args.realesrgan_scale, args.realesrgan_tile,
        explicit_model=args.realesrgan_model, debug=debug
    )

    loading_time = time.time() - loading_start

    # Create streaming video writer at final dimensions
    video_writer = create_streaming_video_writer(output_path, output_width, output_height, fps, debug, with_audio=(audio_path is not None))
    
    batch_size = _get_realesrgan_batch_size(input_height, input_width)
    use_batched = args.realesrgan_tile == 0  # Batching only works without tiling
    if not use_batched:
        batch_size = 1

    mode_str = f"streaming, batch={batch_size}"
    if lanczos_2x:
        mode_str += f", +Lanczos 2x → {output_width}x{output_height}"
    print(f"🎬 Processing {total_frames} frames with {model_name} ({mode_str})")
    if debug and args.realesrgan_tile > 0:
        print(f"🔲 Using tiled processing (tile size: {args.realesrgan_tile})")

    fallback_to_single = False  # Set on OOM, stays for rest of video

    try:
        frames_written = 0
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_time = time.time()

            # Read batch of frames
            batch_frames = []
            for i in range(batch_start, batch_end):
                if video_cap is not None:
                    ret, frame_bgr = video_cap.read()
                    if not ret:
                        if debug:
                            print(f"⚠️ Video read ended at frame {i}/{total_frames}")
                        break
                else:
                    frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
                batch_frames.append(frame_bgr)

            if not batch_frames:
                break

            # Process batch
            try:
                if use_batched and not fallback_to_single and len(batch_frames) > 1:
                    batch_outputs = _batch_enhance(upsampler, batch_frames, args.realesrgan_scale)
                else:
                    batch_outputs = [upsampler.enhance(f, outscale=args.realesrgan_scale)[0] for f in batch_frames]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and not fallback_to_single:
                    print(f"⚠️ Batch OOM (batch={len(batch_frames)}), falling back to single-frame mode")
                    fallback_to_single = True
                    torch.cuda.empty_cache()
                    gc.collect()
                    batch_outputs = [upsampler.enhance(f, outscale=args.realesrgan_scale)[0] for f in batch_frames]
                else:
                    raise

            # Write outputs to video (with optional Lanczos 2x folded in)
            for output in batch_outputs:
                if lanczos_2x:
                    output = cv2.resize(output, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                video_writer.write(output)

            frames_written += len(batch_outputs)

            if _progress:
                _progress.stage('realesrgan', frames_written / total_frames)
            if debug and frames_written % 50 < batch_size:
                print(f"🔄 Processed and wrote {frames_written}/{total_frames} frames")

            inference_time += (time.time() - batch_time)

    finally:
        # Always close video writer and video capture
        video_writer.release()
        if video_cap is not None:
            video_cap.release()

        # Handle audio merging if needed
        if hasattr(video_writer, 'temp_path') and video_writer.temp_path:
            if audio_path and os.path.exists(audio_path):
                try:
                    merge_audio_video(video_writer.temp_path, audio_path, video_writer.final_path, debug)
                    os.unlink(video_writer.temp_path)
                except Exception as e:
                    # If merge fails, just move video without audio
                    shutil.move(video_writer.temp_path, video_writer.final_path)
                    if debug:
                        print(f"⚠️ Audio merge failed, saved video without audio: {e}")
            else:
                shutil.move(video_writer.temp_path, video_writer.final_path)

    if debug:
        print(f"✅ RealESRGAN streaming processing complete")
        print(f"📁 Video saved to: {output_path}")

    return None, loading_time, inference_time  # No tensor returned in streaming mode


def parse_resolution(resolution_str):
    """Parse resolution string to get target dimensions and RealESRGAN scale"""
    resolution_str = resolution_str.lower().strip()
    
    # Common resolution presets
    presets = {
        '1080p': (1920, 1080),
        '1080': (1920, 1080),
        'fhd': (1920, 1080),
        '1440p': (2560, 1440),
        '1440': (2560, 1440),
        '2k': (2560, 1440),
        '4k': (3840, 2160),
        '2160p': (3840, 2160),
        '2160': (3840, 2160),
        'uhd': (3840, 2160),
    }
    
    if resolution_str in presets:
        return presets[resolution_str]
    
    # Try to parse custom resolution (e.g., "1920x1080")
    if 'x' in resolution_str:
        try:
            width, height = map(int, resolution_str.split('x'))
            return (width, height)
        except (ValueError, TypeError):
            pass
    
    raise ValueError(f"Invalid resolution: {resolution_str}. Use formats like '1080p', '4k', or '1920x1080'")


def get_optimal_settings(target_resolution, creative_mode=False):
    """Get optimal settings based on target resolution and creative mode"""
    width, height = target_resolution
    pixels = width * height

    settings = {
        # Base settings that work well for most cases
        'input_max_resolution': 620,
        'max_seedvr2_resolution': 1080,  # SeedVR2 output resolution
        'batch_size': 64,
        'blur_type': 'gaussian',
        'blur_strength': 2.0 if creative_mode else 0.5,  # 2.0 for creative, 0.5 for fast mode
        'preserve_vram': False,

        # Resolution-dependent settings
        'realesrgan_tile': 0,  # Disabled for better performance
        'realesrgan_scale': 2,
        'skip_realesrgan': False,
        'use_lanczos_upscale': False,
        'target_width': width,
        'target_height': height,
    }

    # Adjust settings based on target resolution (SeedVR2 outputs 1280p now)
    if pixels <= 1920 * 1080:  # 1080p or lower
        # SeedVR2 outputs 1280p - close enough for 1080p targets (better quality)
        settings['skip_realesrgan'] = True
        settings['use_lanczos_upscale'] = False  # Output 1280p directly
        settings['realesrgan_scale'] = None
    elif pixels <= 2560 * 1440:  # 1440p
        # 1280p → Lanczos → 1440p (1.125x upscale, fast and good quality)
        settings['skip_realesrgan'] = True
        settings['use_lanczos_upscale'] = True
        settings['realesrgan_scale'] = None
    else:  # 4K (3840x2160) and above
        # 1080p → RealESRGAN 2x → 2160p → Lanczos 2x → 4320p (4x total)
        settings['realesrgan_scale'] = 2
        settings['realesrgan_tile'] = 0  # No tiling needed for 1080p input
        settings['skip_realesrgan'] = False
        settings['use_lanczos_upscale'] = True  # Apply Lanczos 2x after RealESRGAN

    return settings


def parse_arguments():
    """Parse command line arguments for combined pipeline"""
    parser = argparse.ArgumentParser(
        description="AI Video Upscaler - Enhance and upscale videos to HD/4K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mode (SeedVR2 only, minimal blur) - DEFAULT
  python pipeline_upscaler_hypir_fast.py input.mp4 1080p

  # Creative mode (GFPGAN + HyPIR + SeedVR2, more blur)
  python pipeline_upscaler_hypir_fast.py input.mp4 1080p --creative

  # Creative mode with parallel processing (2x speed)
  python pipeline_upscaler_hypir_fast.py input.mp4 4k --creative --hypir-parallel

  # Upscale to 4K (fast mode)
  python pipeline_upscaler_hypir_fast.py input.mp4 4k

  # Upscale to 1440p (2K)
  python pipeline_upscaler_hypir_fast.py input.mp4 1440p

  # Specify output file
  python pipeline_upscaler_hypir_fast.py input.mp4 4k --output enhanced_4k.mp4

  # Creative mode with custom HyPIR prompt
  python pipeline_upscaler_hypir_fast.py input.mp4 1080p --creative --hypir-prompt "high quality, sharp details"

Supported resolutions:
  - 1080p, FHD (1920x1080)
  - 1440p, 2K (2560x1440)  
  - 4K, 2160p, UHD (3840x2160)
  - Custom: WIDTHxHEIGHT
        """
    )
    
    # Required arguments
    parser.add_argument("video", type=str,
                        help="Input video file path")
    parser.add_argument("resolution", type=str,
                        help="Target resolution (e.g., 1080p, 1440p, 4k, 1920x1080)")
    
    # Optional arguments (simple)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: auto-generated)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Don't preserve audio from input video")
    parser.add_argument("--fast", action="store_true",
                        help="Enable fast mode (skip FlashVSR, go directly to SeedVR2). Default: creative mode with FlashVSR")
    parser.add_argument("--no-flashvsr", action="store_true",
                        help="Disable FlashVSR stage (useful for non-Ampere GPUs like T4)")
    parser.add_argument("--creative", action="store_true",
                        help="Force creative mode even if text detected (overrides auto-skip)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    # FlashVSR options
    flashvsr_group = parser.add_argument_group('FlashVSR options')
    flashvsr_group.add_argument("--flashvsr-sparse-ratio", type=float, default=2.0,
                        help="Sparse ratio for attention (1.5 or 2.0, default: 2.0)")
    flashvsr_group.add_argument("--flashvsr-local-range", type=int, default=11, choices=[9, 11],
                        help="Local attention range: 9=sharper details, 11=more stable (default: 11)")
    flashvsr_group.add_argument("--flashvsr-tiled", action="store_true", default=True,
                        help="Enable tiled processing (lower VRAM, slower, default: enabled)")
    flashvsr_group.add_argument("--flashvsr-no-tiled", dest="flashvsr_tiled", action="store_false",
                        help="Disable tiled processing (faster but higher VRAM)")

    # Hidden FlashVSR parameters (auto-configured)
    flashvsr_group.add_argument("--flashvsr-scale", type=float, default=2.0, help=argparse.SUPPRESS)
    flashvsr_group.add_argument("--flashvsr-auto-mode", action="store_true", default=True, help=argparse.SUPPRESS)

    # Text protection options (deprecated - now handled by smart mode)
    text_group = parser.add_argument_group('Text protection options')
    text_group.add_argument("--blur-strength-text", type=float, default=1.0,
                        help=argparse.SUPPRESS)
    text_group.add_argument("--text-coverage-threshold", type=float, default=20.0,
                        help=argparse.SUPPRESS)
    
    # Advanced options (hidden from basic usage)
    advanced = parser.add_argument_group('advanced options')
    advanced.add_argument("--pipeline-mode", type=str, default="all",
                        choices=["gfpgan-only", "seedvr2-only", "realesrgan-only", "gfpgan-seedvr2", "seedvr2-realesrgan", "all"],
                        help=argparse.SUPPRESS)
    advanced.add_argument("--save-intermediate", action="store_true",
                        help="Save intermediate outputs from each stage")
    advanced.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing AI models")
    advanced.add_argument("--cuda-device", type=str, default=None,
                        help="CUDA device(s) to use")

    # SeedVR2 advanced options (exposed for power users)
    seedvr2_advanced = parser.add_argument_group('SeedVR2 advanced options')
    seedvr2_advanced.add_argument("--blocks_to_swap", type=int, default=0,
                        help="BlockSwap: Number of transformer blocks to swap (0-32 for 3B, 0-36 for 7B). Reduces VRAM usage. Requires --dit_offload_device")
    seedvr2_advanced.add_argument("--swap_io_components", action="store_true",
                        help="BlockSwap: Also swap I/O components for extra VRAM savings. Requires --dit_offload_device")
    seedvr2_advanced.add_argument("--dit_offload_device", type=str, default="none",
                        help="Device to offload DiT model when idle: 'none' (default), 'cpu', or GPU ID. Frees VRAM between stages")
    seedvr2_advanced.add_argument("--vae_offload_device", type=str, default="none",
                        help="Device to offload VAE model when idle: 'none' (default), 'cpu', or GPU ID. Frees VRAM between stages")
    seedvr2_advanced.add_argument("--tensor_offload_device", type=str, default="cpu",
                        help="Intermediate tensor storage: 'cpu' (default, save VRAM), 'none' (keep on GPU, faster for short videos), or GPU ID. Auto: GPU for ≤800 frames, CPU for longer.")
    seedvr2_advanced.add_argument("--vae_encode_tiled", action="store_true",
                        help="Enable VAE encode tiling for high resolution (reduces VRAM)")
    seedvr2_advanced.add_argument("--vae_decode_tiled", action="store_true",
                        help="Enable VAE decode tiling for high resolution (reduces VRAM)")
    seedvr2_advanced.add_argument("--vae_encode_tile_size", type=int, default=1024,
                        help="VAE encode tile size in pixels (default: 1024)")
    seedvr2_advanced.add_argument("--vae_decode_tile_size", type=int, default=1024,
                        help="VAE decode tile size in pixels (default: 1024)")
    seedvr2_advanced.add_argument("--vae_encode_tile_overlap", type=int, default=128,
                        help="VAE encode tile overlap in pixels (default: 128)")
    seedvr2_advanced.add_argument("--vae_decode_tile_overlap", type=int, default=128,
                        help="VAE decode tile overlap in pixels (default: 128)")
    seedvr2_advanced.add_argument("--temporal_overlap", type=int, default=0,
                        help="Frames to overlap between batches/GPUs for smooth blending (default: 0)")
    seedvr2_advanced.add_argument("--prepend_frames", type=int, default=0,
                        help="Prepend N reversed frames to reduce start artifacts (auto-removed, default: 0)")
    seedvr2_advanced.add_argument("--color_correction", type=str, default="none",
                        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                        help="Color correction method (default: none)")
    seedvr2_advanced.add_argument("--attention_mode", type=str, default="sdpa",
                        choices=["sdpa", "flash_attn"],
                        help="Attention backend: 'sdpa' (default) or 'flash_attn' (faster, requires package)")
    seedvr2_advanced.add_argument("--compile_dit", action="store_true",
                        help="Enable torch.compile for DiT (20-40%% speedup, requires PyTorch 2.0+)")
    seedvr2_advanced.add_argument("--compile_vae", action="store_true",
                        help="Enable torch.compile for VAE (15-25%% speedup, requires PyTorch 2.0+)")
    seedvr2_advanced.add_argument("--streaming_threshold", type=int, default=1000,
                        help="Frame count threshold for disk streaming mode (default: 1000 frames)")
    seedvr2_advanced.add_argument("--flashvsr_max_frames", type=int, default=800,
                        help="Maximum frames for FlashVSR in creative mode (default: 800 frames)")
    seedvr2_advanced.add_argument("--flashvsr_scale_threshold", type=int, default=500,
                        help="Above this frame count, FlashVSR uses 1x scale (enhancement only) instead of 2x. Default: 500 frames")

    # Hidden technical parameters with smart defaults
    parser.add_argument("--output_format", type=str, default="video", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument("--skip_first_frames", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--load_cap", type=int, default=0, help=argparse.SUPPRESS)
    
    # Model selection (hidden)
    parser.add_argument("--model", type=str, default="seedvr2_ema_3b_fp8_e4m3fn.safetensors", help=argparse.SUPPRESS)
    
    # All technical parameters will be auto-configured
    parser.add_argument("--resolution", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--blur_type", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--blur_strength", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--input_max_resolution", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_seedvr2_resolution", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--preserve_vram", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--realesrgan-scale", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--realesrgan-tile", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--realesrgan-model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--preserve-audio", action="store_true", default=True, help=argparse.SUPPRESS)
    
    # Legacy compatibility
    parser.add_argument("--video_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cuda_device", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model_dir", type=str, default=None, help=argparse.SUPPRESS)
    
    return parser.parse_args()


def process_with_hybrid_realesrgan_4x(frames_input, args, output_path, fps, debug=False, audio_path=None):
    """Process frames with RealESRGAN 2x + Lanczos 2x per frame for fast 4x upscaling.

    Args:
        frames_input: torch.Tensor [T,H,W,C] float16/float32 range [0,1] OR file path string
    """
    import cv2

    loading_start = time.time()

    # Handle both tensor and file path input
    video_cap = None
    frames_np = None
    if isinstance(frames_input, str):
        video_cap = cv2.VideoCapture(frames_input)
        if not video_cap.isOpened():
            raise ValueError(f"Cannot open video for RealESRGAN hybrid: {frames_input}")
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if debug:
            print(f"📁 Reading frames from file: {total_frames} frames, {W}x{H}")
    else:
        frames_np = frames_input.cpu().numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        total_frames = len(frames_np)
        H, W = frames_np[0].shape[:2]

    # Setup RealESRGAN 2x
    upsampler, model_name = _setup_realesrgan(
        args.model_dir, 2, args.realesrgan_tile,
        explicit_model=args.realesrgan_model, debug=debug
    )

    loading_time = time.time() - loading_start
    
    # Get output dimensions (4x total: 2x from RealESRGAN + 2x from Lanczos)
    output_width, output_height = W * 4, H * 4
    
    # Create video writer for final 4x output
    if audio_path:
        temp_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        actual_path = temp_path
    else:
        actual_path = output_path
        
    # Create video writer with codec fallback (stderr suppressed)
    video_writer = open_video_writer(actual_path, output_width, output_height, fps, debug)
    
    # Process frames: RealESRGAN 2x + Lanczos 2x (batched)
    batch_size = _get_realesrgan_batch_size(H, W)
    use_batched = args.realesrgan_tile == 0  # Batching only works without tiling
    if not use_batched:
        batch_size = 1

    print(f"🎬 Processing {total_frames} frames with {model_name} (hybrid 4x, batch={batch_size})")
    inference_start = time.time()

    fallback_to_single = False

    try:
        frames_written = 0
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)

            # Read batch of frames
            batch_frames = []
            for i in range(batch_start, batch_end):
                if video_cap is not None:
                    ret, frame_bgr = video_cap.read()
                    if not ret:
                        if debug:
                            print(f"⚠️ Video read ended at frame {i}/{total_frames}")
                        break
                else:
                    frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
                batch_frames.append(frame_bgr)

            if not batch_frames:
                break

            # Step 1: RealESRGAN 2x (batched)
            try:
                if use_batched and not fallback_to_single and len(batch_frames) > 1:
                    batch_outputs = _batch_enhance(upsampler, batch_frames, 2)
                else:
                    batch_outputs = [upsampler.enhance(f, outscale=2)[0] for f in batch_frames]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and not fallback_to_single:
                    print(f"⚠️ Batch OOM (batch={len(batch_frames)}), falling back to single-frame mode")
                    fallback_to_single = True
                    torch.cuda.empty_cache()
                    gc.collect()
                    batch_outputs = [upsampler.enhance(f, outscale=2)[0] for f in batch_frames]
                else:
                    raise

            # Step 2: Lanczos 2x + write to video
            for realesrgan_output in batch_outputs:
                final_output = cv2.resize(realesrgan_output, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                video_writer.write(final_output)

            frames_written += len(batch_outputs)

            if _progress:
                _progress.stage('realesrgan', frames_written / total_frames)
    finally:
        if video_cap is not None:
            video_cap.release()
        video_writer.release()

    # Handle audio merging if needed
    if audio_path and os.path.exists(audio_path):
        try:
            merge_audio_video(actual_path, audio_path, output_path, debug)
            if actual_path != output_path:
                os.unlink(actual_path)
        except Exception as e:
            # If merge fails, just move video without audio
            if actual_path != output_path:
                shutil.move(actual_path, output_path)
            if debug:
                print(f"⚠️ Audio merge failed, saved video without audio: {e}")
    elif actual_path != output_path:
        shutil.move(actual_path, output_path)
    
    # Clean up
    del upsampler
    torch.cuda.empty_cache()
    
    inference_time = time.time() - inference_start
    
    return None, loading_time, inference_time


def calculate_optimal_seedvr2_resolution(input_width, input_height, max_resolution, target_scale):
    """Calculate optimal SeedVR2 resolution considering final target"""
    # Current longest side
    longest_side = max(input_width, input_height)
    
    # Don't upscale beyond max_resolution in SeedVR2
    if longest_side >= max_resolution:
        return input_width, input_height
    
    # Calculate scale to reach max_resolution
    scale = max_resolution / longest_side
    
    # Round to ensure even dimensions
    new_width = int(input_width * scale)
    new_height = int(input_height * scale)
    
    # Ensure even dimensions for video encoding
    new_width = new_width if new_width % 2 == 0 else new_width + 1
    new_height = new_height if new_height % 2 == 0 else new_height + 1
    
    return new_width, new_height


def lanczos_upscale_video_file(input_video_path, target_width, target_height, output_path, fps, debug=False, audio_path=None):
    """
    Production-ready Lanczos upscaling from video file to video file
    Zero memory accumulation - true streaming
    """
    if debug:
        print(f"🎬 Upscaling from file: {input_video_path}")

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video_path}")

    # Get dimensions
    current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate target dimensions preserving aspect ratio
    current_aspect = current_width / current_height
    target_aspect = target_width / target_height

    if current_aspect > target_aspect:
        final_width = target_width
        final_height = int(target_width / current_aspect)
    else:
        final_height = target_height
        final_width = int(target_height * current_aspect)

    # Ensure even dimensions
    final_width = final_width if final_width % 2 == 0 else final_width + 1
    final_height = final_height if final_height % 2 == 0 else final_height + 1

    if debug:
        print(f"📐 Upscaling {current_width}x{current_height} → {final_width}x{final_height}")

    # Create output writer with audio support
    video_writer = create_streaming_video_writer(output_path, final_width, final_height, fps, debug, with_audio=(audio_path is not None))

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Upscale frame with Lanczos
            upscaled = cv2.resize(frame, (final_width, final_height), interpolation=cv2.INTER_LANCZOS4)
            video_writer.write(upscaled)

            frame_count += 1
            if _progress and total_frames > 0:
                _progress.stage('lanczos', frame_count / total_frames)
            if debug and frame_count % 100 == 0:
                print(f"📝 Processed {frame_count}/{total_frames} frames")

    finally:
        cap.release()
        video_writer.release()

        # Handle audio merging
        if hasattr(video_writer, 'temp_path') and video_writer.temp_path:
            if audio_path and os.path.exists(audio_path):
                try:
                    merge_audio_video(video_writer.temp_path, audio_path, video_writer.final_path, debug)
                    os.unlink(video_writer.temp_path)
                except Exception as e:
                    shutil.move(video_writer.temp_path, video_writer.final_path)
                    if debug:
                        print(f"⚠️ Audio merge failed: {e}")
            else:
                shutil.move(video_writer.temp_path, video_writer.final_path)

    if debug:
        print(f"✅ File streaming complete: {output_path}")

    return None


def lanczos_upscale_frames_streaming(frames_tensor, target_width, target_height, output_path, fps, debug=False, audio_path=None):
    """Production-ready Lanczos upscaling with streaming output - handles any video length"""

    # Handle both tensor and file input
    if isinstance(frames_tensor, str):
        # Input is a video file path (from SeedVR2 temp output)
        return lanczos_upscale_video_file(frames_tensor, target_width, target_height, output_path, fps, debug, audio_path)

    # Original tensor-based processing continues...
    # Convert to numpy
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)

    # Get current dimensions from first frame
    current_height, current_width = frames_np[0].shape[:2]
    current_aspect = current_width / current_height

    # Calculate target dimensions preserving aspect ratio
    target_aspect = target_width / target_height

    if current_aspect > target_aspect:
        # Video is wider, fit to target width
        final_width = target_width
        final_height = int(target_width / current_aspect)
    else:
        # Video is taller or same aspect, fit to target height
        final_height = target_height
        final_width = int(target_height * current_aspect)

    # Ensure even dimensions for video encoding
    final_width = final_width if final_width % 2 == 0 else final_width + 1
    final_height = final_height if final_height % 2 == 0 else final_height + 1

    if debug:
        print(f"\n🔄 Upscaling from {current_width}x{current_height} to {final_width}x{final_height}")
        print(f"📐 Aspect ratio preserved: {current_aspect:.3f} → {final_width/final_height:.3f}")

    # Create streaming video writer
    video_writer = create_streaming_video_writer(output_path, final_width, final_height, fps, debug, with_audio=(audio_path is not None))

    total_frames = len(frames_np)
    if debug:
        print(f"🎬 Processing {total_frames} frames (streaming mode)")

    try:
        # Stream process each frame directly to video
        for i, frame in enumerate(frames_np):
            # Frame is in RGB format, resize with Lanczos
            upscaled = cv2.resize(frame, (final_width, final_height),
                                interpolation=cv2.INTER_LANCZOS4)

            # Convert RGB to BGR for cv2 video writer
            upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)

            # Write directly to video
            video_writer.write(upscaled_bgr)

            if _progress and total_frames > 0:
                _progress.stage('lanczos', (i + 1) / total_frames)
            if debug and (i + 1) % 50 == 0:
                print(f"🔄 Processed and wrote {i + 1}/{total_frames} frames")

    finally:
        # Always close video writer
        video_writer.release()

        # Handle audio merging if needed
        if hasattr(video_writer, 'temp_path') and video_writer.temp_path:
            if audio_path and os.path.exists(audio_path):
                try:
                    merge_audio_video(video_writer.temp_path, audio_path, video_writer.final_path, debug)
                    os.unlink(video_writer.temp_path)
                except Exception as e:
                    # If merge fails, just move video without audio
                    shutil.move(video_writer.temp_path, video_writer.final_path)
                    if debug:
                        print(f"⚠️ Audio merge failed, saved video without audio: {e}")
            else:
                shutil.move(video_writer.temp_path, video_writer.final_path)

    if debug:
        print(f"✅ Streaming upscaling complete")
        print(f"📁 Video saved to: {output_path}")

    return None  # No tensor returned in streaming mode


def main():
    """Main pipeline function"""
    args = parse_arguments()
    
    # Handle legacy video_path argument
    if hasattr(args, 'video_path') and args.video_path:
        args.video = args.video_path
    
    # Parse target resolution
    try:
        target_width, target_height = parse_resolution(args.resolution)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Get optimal settings based on target resolution and creative mode
    # Default is creative mode (with FlashVSR), unless --fast or --no-flashvsr is specified
    # --creative forces creative mode even if text/long video detected (no auto-skip)
    if args.no_flashvsr:
        creative_mode = False
    elif not FLASHVSR_AVAILABLE:
        creative_mode = False
        print("⚠️ FlashVSR not available - running without it")
    else:
        creative_mode = not args.fast  # Default: True (creative mode)
    force_creative = args.creative  # --creative means force creative (override auto-skip)
    optimal_settings = get_optimal_settings((target_width, target_height), creative_mode=creative_mode)

    # Apply optimal settings if not explicitly set by user
    if args.batch_size is None:
        args.batch_size = optimal_settings['batch_size']
    if args.blur_type is None:
        args.blur_type = optimal_settings['blur_type']
    if args.blur_strength is None:
        args.blur_strength = optimal_settings['blur_strength']
    if args.input_max_resolution is None:
        args.input_max_resolution = optimal_settings['input_max_resolution']
    if args.max_seedvr2_resolution is None:
        args.max_seedvr2_resolution = optimal_settings['max_seedvr2_resolution']
    if args.realesrgan_scale is None:
        args.realesrgan_scale = optimal_settings['realesrgan_scale']
    if args.realesrgan_tile is None:
        args.realesrgan_tile = optimal_settings['realesrgan_tile']
    if args.preserve_vram is None:
        args.preserve_vram = optimal_settings['preserve_vram']
    
    # Store additional settings for pipeline logic
    args.skip_realesrgan = optimal_settings['skip_realesrgan']
    args.use_lanczos_upscale = optimal_settings['use_lanczos_upscale']
    args.target_width = optimal_settings['target_width']
    args.target_height = optimal_settings['target_height']
    
    # Set model directory defaults
    if not hasattr(args, 'model_dir') or args.model_dir is None:
        # Try common locations — check that the directory actually contains model files
        model_dir_candidates = [
            "seedvr2_models",
            "models",
            "/content/ComfyUI-SeedVR2_VideoUpscaler/models",
            os.path.join(os.path.dirname(__file__), "models"),
        ]
        for candidate in model_dir_candidates:
            if os.path.isdir(candidate) and any(
                f.endswith(('.safetensors', '.pth'))
                for f in os.listdir(candidate)
            ):
                args.model_dir = candidate
                break
        if not hasattr(args, 'model_dir') or args.model_dir is None:
            args.model_dir = "models"  # Fallback
    
    # Handle audio preservation flag
    preserve_audio = args.preserve_audio and not args.no_audio
    
    # Simple start message
    print(f"\n🎬 Processing: {Path(args.video).name}")
    print(f"🎯 Target: {target_width}x{target_height}")
    print("⏳ Loading...")

    # Store original debug setting and disable detailed output
    original_debug = args.debug
    args.debug = False

    # Suppress warnings in main process when not debugging
    if not original_debug:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
    else:
        os.environ.pop("TQDM_DISABLE", None)
    
    # Extract audio if needed
    audio_path = None
    if preserve_audio:
        audio_path = tempfile.NamedTemporaryFile(suffix='.aac', delete=False).name
        try:
            has_audio = extract_audio(args.video, audio_path, original_debug)
        except Exception as e:
            print(f"⚠️ Audio extraction failed: {e}")
            has_audio = False
        if not has_audio:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
            audio_path = None
            if original_debug:
                print("⚠️ No audio track found, proceeding without audio")
    
    # Check GPU memory availability
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if args.debug:
            print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 Total GPU memory: {total_memory:.1f}GB")
        
        # Warn if memory might be insufficient
        if args.pipeline_mode == "both" and total_memory < 32:
            print(f"⚠️ Warning: {total_memory:.1f}GB GPU memory detected. For best results with both models:")
            print(f"   - Consider using --preserve_vram")
            print(f"   - Use --realesrgan-tile 256 or 512")
            print(f"   - Monitor memory usage with --debug")
    
    if args.debug:
        print("\n📋 Pipeline Configuration:")
        print(f"   Mode: {args.pipeline_mode}")
        print(f"   FlashVSR auto-mode: {args.flashvsr_auto_mode}")
        print(f"   FlashVSR scale: {args.flashvsr_scale}x")
        print(f"   SeedVR2 max resolution: {args.max_seedvr2_resolution}p")
        print(f"   RealESRGAN scale: {args.realesrgan_scale}x")
        print(f"   RealESRGAN tiling: {args.realesrgan_tile if args.realesrgan_tile > 0 else 'disabled'}")
        print()
    
    try:
        # Stage 1: Extract and resize frames in single pass (memory efficient)
        start_time = time.time()

        frames_tensor, original_fps = extract_and_resize_frames(
            video_path=args.video,
            max_dimension=620,
            skip_first_frames=args.skip_first_frames,
            load_cap=args.load_cap if args.load_cap > 0 else None,
            debug=original_debug
        )

        extraction_time = time.time() - start_time
        if args.debug:
            print(f"⏱️ Extraction+resize time: {extraction_time:.2f}s")
            print(f"📊 Frames: {frames_tensor.shape}")
        
        # Initialize timing variables
        flashvsr_time = 0

        seedvr2_time = 0
        seedvr2_loading_time = 0
        seedvr2_inference_time = 0

        realesrgan_time = 0
        realesrgan_loading_time = 0
        realesrgan_inference_time = 0

        lanczos_time = 0
        save_time = 0
        
        # Get input dimensions
        T, H, W, C = frames_tensor.shape
        
        # Calculate optimal SeedVR2 resolution
        if args.pipeline_mode in ["all", "seedvr2-only", "gfpgan-seedvr2", "seedvr2-realesrgan"]:
            # SeedVR2 output should be capped at max_seedvr2_resolution (1080p)
            # This ensures SeedVR2 respects the 1280p limit regardless of input size
            args.resolution = args.max_seedvr2_resolution
            if original_debug:
                print(f"📐 SeedVR2 target resolution: {args.max_seedvr2_resolution}p")
        
        current_frames = frames_tensor

        # Initialize text detection variables
        text_regions = {}
        text_coverage = 0.0

        # Get total frame count for smart decisions
        total_frames = frames_tensor.shape[0]

        # OPTIMIZATION: Skip FlashVSR for long videos
        # Long videos are better suited for GFPGAN + SeedVR2 (faster, less VRAM)
        flashvsr_max_frames = getattr(args, 'flashvsr_max_frames', 800)  # Use arg or default to 800
        skip_text_detection = False  # Flag to skip text detection entirely

        if creative_mode and not force_creative and total_frames > flashvsr_max_frames:
            if original_debug:
                print(f"\n⚡ Video too long ({total_frames} frames > {flashvsr_max_frames}) - skipping FlashVSR")
                print(f"💡 Long video: Direct to SeedVR2 (no text detection needed)")
            creative_mode = False
            args.blur_strength = 0.5  # Long videos always use mild blur (0.5)
            skip_text_detection = True

        # Smart creative mode: Detect text upfront to decide if creative mode helps
        # Only do this for short/medium videos where FlashVSR is viable
        # Skip detection entirely if force_creative (--creative flag)
        elif creative_mode and not force_creative and not skip_text_detection:
            if original_debug:
                print("🔍 Running text detection to check if creative mode is suitable...")

            text_regions, text_coverage = detect_text_in_video(
                args.video,
                max_frames_to_check=None,  # Check all frames
                debug=original_debug,
                model_dir=args.model_dir,
                script_dir=script_dir
            )

            # Calculate text frame percentage (total_frames already set above)
            text_frame_count = len(text_regions)
            text_frame_percentage = (text_frame_count / total_frames) * 100 if total_frames > 0 else 0

            if original_debug:
                print(f"📊 Text detection results:")
                print(f"   Frames with text: {text_frame_count}/{total_frames} ({text_frame_percentage:.1f}%)")
                print(f"   Avg text coverage: {text_coverage:.1f}%")

            # RULE 1: Auto-skip creative if lots of small/medium text (subtitles/UI)
            # (Never disable if --force-creative is used)
            if text_frame_percentage >= 40 and text_coverage < 11:
                if not force_creative:
                    creative_mode = False
                    # Text detected - use mild blur to preserve text clarity
                    args.blur_strength = 0.5
                    if original_debug:
                        print(f"⚡ Auto-switching to FAST mode (text detected - {text_coverage:.1f}% coverage)")
                        print(f"   Using mild blur (0.5) to preserve text clarity")
                elif original_debug:
                    print(f"🔒 Force-creative mode: Keeping FlashVSR enabled despite text detection")

            # RULE 2: Check for people if text is present (any coverage size)
            # Skip FlashVSR if ≥40% frames have text BUT no people detected
            # (likely game UI, screencasts, logos - no human content to enhance)
            elif creative_mode and text_frame_percentage >= 40:
                if original_debug:
                    print("🔍 Text detected - checking for people in video...")

                people_detected = detect_persons_in_video(
                    args.video,
                    confidence_threshold=0.5,
                    max_frames_to_check=30,
                    debug=original_debug,
                    model_dir=args.model_dir
                )

                if not people_detected and not force_creative:
                    creative_mode = False
                    # Text detected + no people - use mild blur to preserve text
                    args.blur_strength = 0.5
                    if original_debug:
                        print(f"⚡ Auto-switching to FAST mode (text present but no people detected)")
                        print(f"   Using mild blur (0.5) to preserve text clarity")
                elif not people_detected and force_creative and original_debug:
                    print(f"🔒 Force-creative mode: Keeping FlashVSR enabled despite no people detected")

            # RULE 3: If creative mode disabled but NO text detected (clean video)
            # Use mild blur (0.5)
            elif not creative_mode and text_frame_percentage < 40:
                args.blur_strength = 0.5
                if original_debug:
                    print(f"✨ Fast mode with clean video (minimal text: {text_frame_percentage:.1f}%)")
                    print(f"   Using mild blur (0.5)")

        # Setup unified progress bar after all mode decisions
        if not original_debug:
            global _progress
            weights = []
            will_flashvsr = creative_mode and args.pipeline_mode in ["all", "gfpgan-only", "gfpgan-seedvr2"]
            will_seedvr2 = args.pipeline_mode in ["all", "seedvr2-only", "gfpgan-seedvr2", "seedvr2-realesrgan"]
            will_realesrgan = args.pipeline_mode in ["all", "realesrgan-only", "seedvr2-realesrgan"] and not args.skip_realesrgan
            will_lanczos = args.pipeline_mode in ["all", "realesrgan-only", "seedvr2-realesrgan"] and (args.skip_realesrgan or args.use_lanczos_upscale)
            if will_flashvsr: weights.append(('flashvsr', 15))
            if will_seedvr2: weights.append(('seedvr2', 45))
            if will_realesrgan: weights.append(('realesrgan', 37))
            if will_lanczos and not will_realesrgan: weights.append(('lanczos', 40))
            elif will_lanczos: weights.append(('lanczos', 3))
            if not weights: weights = [('processing', 100)]
            progress = ProgressBar()
            total_w = sum(w for _, w in weights)
            cumulative = 0
            for name, w in weights:
                s = round(cumulative * 100 / total_w)
                cumulative += w
                e = round(cumulative * 100 / total_w)
                progress.ranges[name] = (s, e)
            _progress = progress
            progress.update(0)

        # Stage 2: FlashVSR Video Super-Resolution (only if creative mode enabled)
        if creative_mode and args.pipeline_mode in ["all", "gfpgan-only", "gfpgan-seedvr2"]:
            # Text detection already done in smart mode check above (if needed)
            # Skip text detection if force_creative (--creative flag)
            if not text_regions and not force_creative:
                text_regions, text_coverage = detect_text_in_video(
                    args.video,
                    max_frames_to_check=None,  # Check all frames
                    debug=original_debug,
                    model_dir=args.model_dir,
                    script_dir=script_dir
                )

            # Run FlashVSR (replaces GFPGAN+HyPIR as one unit)
            flashvsr_start = time.time()
            flashvsr_result, flashvsr_loading_time, flashvsr_inference_time = process_with_flashvsr(current_frames, args, args.debug, text_regions)
            flashvsr_time = time.time() - flashvsr_start
            if _progress:
                _progress.stage('flashvsr', 1.0)

            if args.debug:
                print(f"⏱️ FlashVSR total time: {flashvsr_time:.2f}s (loading: {flashvsr_loading_time:.2f}s, inference: {flashvsr_inference_time:.2f}s)")
                print(f"📊 FlashVSR output shape: {flashvsr_result.shape}")
                allocated, reserved = get_gpu_memory_info()
                print(f"📊 GPU memory after FlashVSR: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

            current_frames = flashvsr_result

            # Save FlashVSR intermediate if requested
            if args.save_intermediate or args.pipeline_mode == "gfpgan-only":
                base_name = Path(args.video).stem
                intermediate_path = args.output or f"output/{base_name}_flashvsr.mp4"
                if args.pipeline_mode == "gfpgan-only":
                    intermediate_path = args.output or f"output/{base_name}_flashvsr_enhanced.mp4"

                os.makedirs(os.path.dirname(intermediate_path) or "output", exist_ok=True)

                if args.output_format == "png":
                    base_name = Path(args.video).stem + "_flashvsr"
                    save_frames_to_png(current_frames, intermediate_path, base_name, args.debug)
                else:
                    save_frames_to_video(current_frames, intermediate_path, original_fps, args.debug)

                if args.save_intermediate:
                    print(f"💾 Intermediate FlashVSR output saved: {intermediate_path}")

            # Clear memory before next stage
            if args.pipeline_mode in ["all", "gfpgan-seedvr2"]:
                del flashvsr_result
                clear_gpu_memory(preserve_data=True)

            # Mark that FlashVSR ran
            flashvsr_ran = True
        else:
            # FlashVSR skipped (fast mode or text-heavy video)
            flashvsr_ran = False

        # Stage 3: SeedVR2 Processing (using new implementation)
        if args.pipeline_mode in ["all", "seedvr2-only", "gfpgan-seedvr2", "seedvr2-realesrgan"]:
            seedvr2_start = time.time()

            # Apply blur and resize for SeedVR2 input (still needed for optimal processing)
            # The pipeline controls input resolution separately from SeedVR2's output resolution
            # SeedVR2 works best with 620p input, then upscales to target resolution (1280p)
            current_frames = apply_blur_and_resize_to_frames(
                current_frames,
                args.blur_type,
                args.blur_strength,
                args.input_max_resolution,  # This is 620p (input to SeedVR2)
                args.debug
            )

            # Free original extracted frames — no longer needed after blur+resize
            del frames_tensor
            gc.collect()

            # Import here to avoid loading if not needed
            from src.utils.downloads import download_weight

            # Parse GPU list for SeedVR2
            cuda_device = getattr(args, 'cuda_device', None)
            device_list = [d.strip() for d in str(cuda_device).split(',') if d.strip()] if cuda_device else ["0"]

            # Track loading time
            loading_start = time.time()
            # Download model if needed (using new CLI's download function)
            download_weight(dit_model=args.model, vae_model=None, model_dir=args.model_dir, debug=None)
            seedvr2_loading_time = time.time() - loading_start

            # Create SeedVR2 args from pipeline args (with new parameters)
            seedvr2_args = create_seedvr2_args_from_pipeline(args, model_dir=args.model_dir)

            # Override with pipeline-specific settings
            # Failsafe: Ensure max_seedvr2_resolution is always set
            if not hasattr(args, 'max_seedvr2_resolution') or args.max_seedvr2_resolution is None:
                args.max_seedvr2_resolution = 1080
                if original_debug:
                    print(f"⚠️ WARNING: max_seedvr2_resolution was None, defaulting to 1080p")

            # Set both resolution parameters to 1080p
            seedvr2_args.resolution = args.max_seedvr2_resolution  # Target resolution (1080p)
            seedvr2_args.max_resolution = args.max_seedvr2_resolution  # Also set max_resolution to same value

            # Check frame count for adaptive settings
            frame_count = current_frames.shape[0]
            streaming_threshold = getattr(args, 'streaming_threshold', 800)

            # Dynamic tensor_offload_device based on video length
            # Short videos (<= 800 frames): Keep tensors on GPU for maximum speed
            # Long videos (> 800 frames): Offload to CPU to prevent OOM and save VRAM
            if not hasattr(args, 'tensor_offload_device') or args.tensor_offload_device == 'cpu':
                if frame_count <= streaming_threshold:
                    seedvr2_args.tensor_offload_device = 'none'  # GPU - faster
                    if original_debug:
                        print(f"⚡ Short video ({frame_count} frames): Keeping tensors on GPU for speed")
                else:
                    seedvr2_args.tensor_offload_device = 'cpu'  # CPU - safer for long videos
                    if original_debug:
                        print(f"💾 Long video ({frame_count} frames): Offloading tensors to CPU for memory safety")
            else:
                # User explicitly set tensor_offload_device, respect it
                if original_debug:
                    print(f"🔧 Using user-specified tensor_offload_device: {args.tensor_offload_device}")

            if original_debug:
                print(f"DEBUG: args.resolution = {getattr(args, 'resolution', 'NOT SET')}")
                print(f"DEBUG: args.max_seedvr2_resolution = {getattr(args, 'max_seedvr2_resolution', 'NOT SET')}")
                print(f"DEBUG: seedvr2_args.resolution BEFORE = {seedvr2_args.resolution}")
                print(f"DEBUG: seedvr2_args.max_resolution BEFORE = {seedvr2_args.max_resolution}")
                print(f"DEBUG: seedvr2_args.resolution AFTER = {seedvr2_args.resolution}")
                print(f"DEBUG: seedvr2_args.max_resolution AFTER = {seedvr2_args.max_resolution}")
                print(f"🔧 SeedVR2 settings: resolution={seedvr2_args.resolution}p, max_resolution={seedvr2_args.max_resolution}p, "
                      f"batch_size={seedvr2_args.batch_size}, temporal_overlap={seedvr2_args.temporal_overlap}, "
                      f"tensor_offload={seedvr2_args.tensor_offload_device}, "
                      f"color_correction={seedvr2_args.color_correction}")

            # Process with adaptive SeedVR2 implementation for long videos
            inference_start = time.time()
            seedvr2_temp_file = None  # Track temp file for cleanup
            if frame_count > streaming_threshold:  # Long video threshold
                if original_debug:
                    print(f"📊 Long video detected ({frame_count} frames) - using disk streaming")

                seedvr2_result, seedvr2_temp_file = process_seedvr2_adaptive(
                    current_frames, args, original_fps, device_list, seedvr2_args, original_debug,
                    threshold=streaming_threshold
                )

                if seedvr2_temp_file:
                    current_frames = seedvr2_temp_file  # Pass file path instead of tensor
            else:
                # Short video - use original in-memory processing
                seedvr2_result = inference_cli_new._gpu_processing(current_frames, device_list, seedvr2_args)
                current_frames = seedvr2_result

            seedvr2_inference_time = time.time() - inference_start
            if _progress:
                _progress.stage('seedvr2', 1.0)

            seedvr2_time = time.time() - seedvr2_start
            if args.debug:
                print(f"⏱️ SeedVR2 total time: {seedvr2_time:.2f}s (loading: {seedvr2_loading_time:.2f}s, inference: {seedvr2_inference_time:.2f}s)")
                if seedvr2_result is not None:
                    print(f"📊 SeedVR2 output shape: {seedvr2_result.shape}")
                allocated, reserved = get_gpu_memory_info()
                print(f"📊 GPU memory after SeedVR2: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
            # Save intermediate if requested
            if args.save_intermediate or args.pipeline_mode == "seedvr2-only":
                base_name = Path(args.video).stem
                intermediate_path = args.output or f"output/{base_name}_seedvr2.mp4"
                if args.pipeline_mode == "seedvr2-only":
                    intermediate_path = args.output or f"output/{base_name}_seedvr2_enhanced.mp4"
                
                os.makedirs(os.path.dirname(intermediate_path) or "output", exist_ok=True)
                
                # Handle both tensor and file outputs
                if isinstance(current_frames, str):
                    # Copy temp file to output
                    shutil.copy2(current_frames, intermediate_path)
                else:
                    if args.output_format == "png":
                        base_name = Path(args.video).stem + "_seedvr2"
                        save_frames_to_png(current_frames, intermediate_path, base_name, args.debug)
                    else:
                        # Use streaming save for long videos
                        if current_frames.shape[0] > getattr(args, 'streaming_threshold', 800):
                            save_tensor_to_video_streaming(current_frames, intermediate_path, original_fps, args.debug)
                        else:
                            save_frames_to_video(current_frames, intermediate_path, original_fps, args.debug)
                
                if args.save_intermediate:
                    print(f"💾 Intermediate SeedVR2 output saved: {intermediate_path}")
            
            # If continuing to RealESRGAN, optimize memory for next stage
            if args.pipeline_mode in ["all", "seedvr2-realesrgan"]:
                # Only optimize if we have tensor data
                if not isinstance(current_frames, str):
                    if args.debug:
                        allocated, reserved = get_gpu_memory_info()
                        print(f"📊 GPU memory before optimization: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

                    # Full memory cleanup — evict SeedVR2 models before RealESRGAN loads
                    if 'seedvr2_result' in locals() and seedvr2_result is not None:
                        del seedvr2_result
                    clear_gpu_memory(preserve_data=False)

                    if args.debug:
                        allocated, reserved = get_gpu_memory_info()
                        print(f"📊 GPU memory after optimization: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
                        print(f"🎯 Frames kept in GPU memory: {current_frames.shape}")
        
        # Stage 4: Final Upscaling (RealESRGAN or Lanczos)
        if original_debug:
            print(f"\n🎯 Stage 4: Final Upscaling")
            print(f"   pipeline_mode: {args.pipeline_mode}")
            print(f"   skip_realesrgan: {args.skip_realesrgan}")
            print(f"   use_lanczos_upscale: {args.use_lanczos_upscale}")
            print(f"   realesrgan_scale: {args.realesrgan_scale}")
            print(f"   target: {target_width}x{target_height}")

        if args.pipeline_mode in ["all", "realesrgan-only", "seedvr2-realesrgan"]:
            if original_debug:
                print(f"✅ Pipeline mode matches - will run RealESRGAN or Lanczos")

            # Check if we should skip RealESRGAN (use Lanczos only)
            if args.skip_realesrgan:
                if original_debug:
                    print(f"🔄 Using Lanczos upscaling (RealESRGAN skipped)")
                upscale_start = time.time()

                # Generate output path for streaming
                if args.output is None:
                    os.makedirs('output', exist_ok=True)
                    base_name = Path(args.video).stem

                    # Simple suffix based on target resolution
                    res_suffix = f"_{target_width}x{target_height}"
                    if target_width == 1920 and target_height == 1080:
                        res_suffix = "_1080p"
                    elif target_width == 2560 and target_height == 1440:
                        res_suffix = "_1440p"
                    elif target_width == 3840 and target_height == 2160:
                        res_suffix = "_4K"

                    streaming_output_path = f"output/{base_name}_upscaled{res_suffix}.mp4"
                else:
                    streaming_output_path = args.output

                # Use streaming Lanczos upscaling (production-ready)
                # Now handles both tensor and file input
                lanczos_upscale_frames_streaming(
                    current_frames,
                    args.target_width,
                    args.target_height,
                    streaming_output_path,
                    original_fps,
                    original_debug,
                    audio_path
                )

                # Cleanup temp file if it exists
                if isinstance(current_frames, str) and os.path.exists(current_frames):
                    try:
                        os.unlink(current_frames)
                        if original_debug:
                            print(f"🗑️ Cleaned up temp file: {current_frames}")
                    except OSError as e:
                        print(f"⚠️ Failed to clean up temp file {current_frames}: {e}")

                realesrgan_time = time.time() - upscale_start
                realesrgan_loading_time = 0
                realesrgan_inference_time = realesrgan_time
                output_already_saved = True  # Streaming saved the output

                if original_debug:
                    print(f"⏱️ Upscaling time: {realesrgan_time:.2f}s")

            else:
                if original_debug:
                    print(f"🎨 Using RealESRGAN {args.realesrgan_scale}x upscaling")

                # Use RealESRGAN for higher resolutions
                realesrgan_start = time.time()
                
                # Always use streaming mode for RealESRGAN - it's better for all video lengths
                # Streaming mode: Zero memory accumulation, handles any video length, better performance
                # Generate output path for streaming
                if args.output is None:
                    os.makedirs('output', exist_ok=True)
                    base_name = Path(args.video).stem
                    
                    # Simple suffix based on target resolution
                    res_suffix = f"_{target_width}x{target_height}"
                    if target_width == 1920 and target_height == 1080:
                        res_suffix = "_1080p"
                    elif target_width == 2560 and target_height == 1440:
                        res_suffix = "_1440p"
                    elif target_width == 3840 and target_height == 2160:
                        res_suffix = "_4K"
                    
                    streaming_output_path = f"output/{base_name}_upscaled{res_suffix}.mp4"
                else:
                    streaming_output_path = args.output

                # Smart scaling: Use RealESRGAN 2x + Lanczos 2x for 4x requests (per-frame processing)
                # Both functions accept tensor or file path — file path = zero RAM accumulation
                if isinstance(current_frames, str) and original_debug:
                    print(f"📁 Processing RealESRGAN from temp file: {current_frames}")

                if args.realesrgan_scale == 4:
                    final_result, realesrgan_loading_time, realesrgan_inference_time = process_with_hybrid_realesrgan_4x(
                        current_frames, args, streaming_output_path, original_fps, args.debug, audio_path
                    )
                else:
                    # Fold Lanczos 2x into the RealESRGAN write loop (no separate disk round-trip)
                    final_result, realesrgan_loading_time, realesrgan_inference_time = process_with_realesrgan_streaming(
                        current_frames, args, streaming_output_path, original_fps, args.debug, audio_path,
                        lanczos_2x=args.use_lanczos_upscale
                    )

                # Cleanup temp file if input was a file path
                if isinstance(current_frames, str) and os.path.exists(current_frames):
                    try:
                        os.unlink(current_frames)
                        if original_debug:
                            print(f"🗑️ Cleaned up temp file: {current_frames}")
                    except OSError as e:
                        print(f"⚠️ Failed to clean up temp file {current_frames}: {e}")

                realesrgan_time = time.time() - realesrgan_start
                if args.debug:
                    print(f"⏱️ RealESRGAN total time: {realesrgan_time:.2f}s (loading: {realesrgan_loading_time:.2f}s, inference: {realesrgan_inference_time:.2f}s)")
                    allocated, reserved = get_gpu_memory_info()
                    print(f"📊 GPU memory after RealESRGAN: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

                # Free source frames — RealESRGAN has already streamed to disk
                del current_frames
                gc.collect()
                torch.cuda.empty_cache()

                # Lanczos 2x is now folded into process_with_realesrgan_streaming (no separate step)

                # Mark that we've already saved the final output
                output_already_saved = True

        else:
            if original_debug:
                print(f"⏭️ Skipping Stage 4 - pipeline_mode '{args.pipeline_mode}' doesn't include RealESRGAN")
            output_already_saved = False
        

        # Stage 5: Save final output (if not already saved via streaming)
        save_time = 0
        if not output_already_saved:
            # Generate output path
            if args.output is None:
                os.makedirs('output', exist_ok=True)
                base_name = Path(args.video).stem
                
                # Simple suffix based on target resolution
                res_suffix = f"_{target_width}x{target_height}"
                if target_width == 1920 and target_height == 1080:
                    res_suffix = "_1080p"
                elif target_width == 2560 and target_height == 1440:
                    res_suffix = "_1440p"
                elif target_width == 3840 and target_height == 2160:
                    res_suffix = "_4K"
                
                if args.output_format == "png":
                    output_path = f"output/{base_name}_upscaled{res_suffix}"
                else:
                    output_path = f"output/{base_name}_upscaled{res_suffix}.mp4"
            else:
                output_path = args.output
            
            # Save output
            save_start = time.time()
            if args.output_format == "png":
                base_name = Path(args.video).stem + "_final"
                save_frames_to_png(current_frames, output_path, base_name, args.debug)
            else:
                # Use audio-enabled save function for 1080p
                save_frames_to_video_with_audio(current_frames, output_path, original_fps, args.debug, audio_path)


            save_time = time.time() - save_start
        else:
            output_path = streaming_output_path  # Use the path from streaming
        
        # Simple completion message
        if _progress:
            _progress.complete()

        total_time = time.time() - start_time
        print(f"✅ Completed in {int(total_time)} seconds")
        print(f"📁 Output: {output_path}")

        if original_debug:
            print(f"\n📊 Timing breakdown:")
            print(f"   Extraction+resize: {extraction_time:.1f}s")
            if flashvsr_time > 0:
                print(f"   FlashVSR:          {flashvsr_time:.1f}s")
            if seedvr2_time > 0:
                print(f"   SeedVR2:           {seedvr2_time:.1f}s")
            if realesrgan_time > 0:
                print(f"   RealESRGAN:        {realesrgan_time:.1f}s")
            if lanczos_time > 0:
                print(f"   Lanczos upscale:   {lanczos_time:.1f}s")
            if save_time > 0:
                print(f"   Save output:       {save_time:.1f}s")
            print(f"   Total:             {total_time:.1f}s")

        # Restore debug setting
        args.debug = original_debug
        
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        _progress = None
        # Cleanup all temp files
        # Clean up seedvr2 temp file if it exists
        if 'seedvr2_temp_file' in locals() and seedvr2_temp_file and os.path.exists(seedvr2_temp_file):
            try:
                os.unlink(seedvr2_temp_file)
                if original_debug:
                    print(f"🗑️ Cleaned up SeedVR2 temp file: {seedvr2_temp_file}")
            except OSError as e:
                print(f"⚠️ Failed to clean up SeedVR2 temp file {seedvr2_temp_file}: {e}")

        # Clean up audio temp file if exists
        if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
                if original_debug:
                    print(f"🗑️ Cleaned up audio temp file: {audio_path}")
            except OSError as e:
                print(f"⚠️ Failed to clean up audio temp file {audio_path}: {e}")

        # GPU memory automatically freed
        clear_gpu_memory(preserve_data=False)


if __name__ == "__main__":
    main()