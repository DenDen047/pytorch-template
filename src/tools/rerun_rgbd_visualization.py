import argparse
import os
import sys

import cv2
import numpy as np
import rerun as rr
from loguru import logger
from tqdm import tqdm


def visualize_rgb_depth(rgb_image, depth_map, focal_length=None, timestamp=None):
    """Visualize RGB and depth images with 3D visualization using Rerun's Pinhole camera model.

    Args:
    ----
        rgb_image: RGB image (H, W, 3)
        depth_map: Depth map (H, W)
        focal_length: Camera focal length in pixel
        frame_name: Name for this frame in Rerun
        timestamp: Optional timestamp in seconds for timeline visualization

    """
    # Set the timeline timestamp if provided
    if timestamp is not None:
        rr.set_time_seconds("frame_time", timestamp)

    # Define the camera parameters using Pinhole model
    h, w = depth_map.shape
    rr.log(
        "world/camera/image",
        rr.Pinhole(
            resolution=[w, h],
            focal_length=0.7 * w if focal_length is not None else focal_length,  # Using the same focal length as the official example
        ),
    )

    # Log RGB image with compression
    rr.log("world/camera/image/rgb", rr.Image(rgb_image).compress(jpeg_quality=95))

    # Log depth image
    # Note: You might need to adjust DEPTH_IMAGE_SCALING based on your depth units
    DEPTH_IMAGE_SCALING = 1.0  # Adjust if your depth is not in meters
    rr.log(
        "world/camera/image/depth", rr.DepthImage(depth_map, meter=DEPTH_IMAGE_SCALING)
    )

    # Reset timeline
    rr.reset_time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize depth maps using Rerun")
    parser.add_argument(
        "--depth_file",
        type=str,
        required=True,
        help="Path to the depth maps NPZ or NPY file generated by run_video_depth_anything",
    )
    parser.add_argument(
        "--image_or_video",
        type=str,
        required=True,
        help="Path to the original image or video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/rerun_visualize",
        help="Output directory for visualization files",
    )
    parser.add_argument(
        "--invert_depth",
        action="store_true",
        help="If the depth map is inverted, set this flag",
    )
    parser.add_argument(
        "--depth_scaling",
        type=float,
        default=1.0,
        help="Depth scaling factor",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run Rerun in interactive mode instead of recording",
    )

    # Add Rerun-specific arguments
    rr.script_add_args(parser)
    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(f"{args.output_dir}/log.txt", level="DEBUG", rotation="10 MB")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Rerun
    recording_id = os.path.basename(args.depth_file).split(".")[0]
    rr.init(recording_id, spawn=args.interactive)

    # Load depth maps
    logger.info(f"Loading depth maps from {args.depth_file}")
    if args.depth_file.endswith(".npz"):
        with np.load(args.depth_file) as data:
            depth_maps = data["depth"]
    elif args.depth_file.endswith(".npy"):
        with open(args.depth_file, "rb") as f:
            depth_maps = np.load(f)
    if args.invert_depth:
        depth_maps = 1.0 / depth_maps
    depth_maps = depth_maps * args.depth_scaling
    logger.info(f"Loaded depth maps: {depth_maps.shape}")

    if args.image_or_video.endswith(".mp4"):  # Video mode
        assert len(depth_maps.shape) == 3, "Depth map must be 3D for video mode"
        # Load video
        cap = cv2.VideoCapture(args.image_or_video)
        assert depth_maps.shape[0] == cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        ), "Depth map and video must have the same number of frames"
        # get video info
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        duration = n_frames / fps
        # visualize
        for i in tqdm(range(n_frames), desc="Visualizing video"):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {i}")
                break
            rgb_image = frame
            depth_map = depth_maps[i]
            visualize_rgb_depth(rgb_image, depth_map, timestamp=i / fps)
        cap.release()
    else:  # Image mode
        assert len(depth_maps.shape) == 2, "Depth map must be 2D for image mode"
        # Load RGB image
        rgb_image = cv2.imread(args.image_or_video)

        # Use the simplified visualization without 3D point cloud
        visualize_rgb_depth(
            rgb_image=rgb_image,
            depth_map=depth_maps,
            timestamp=0,
        )

    # If not in interactive mode, save the recording
    if not args.interactive:
        logger.info(f"Saving Rerun recording to {args.output_dir}/rerun_recording.rrd")
        rr.save(f"{args.output_dir}/rerun_recording.rrd")
        logger.info("Visualization complete")
    else:
        logger.info("Interactive visualization running. Press Ctrl+C to exit.")
        # Keep the program running for interactive visualization
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting interactive visualization")

    logger.info(f"All output saved to {args.output_dir}")
