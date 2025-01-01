from src.core.gradient import compute_gradients
from src.core.pyramid import build_pyramid
from src.core.flow_estimation import lucas_kanade_flow
from src.utils.io import load_frames, save_output, write_video
from src.utils.visualization import visualize_flow

import argparse
import cv2
import numpy as np

def parse_args():
    """
    Parse command-line arguments for optical flow script.
    """
    parser = argparse.ArgumentParser( description="Run optical flow on a given video." )

    parser.add_argument(
        "--input",
        type=str,
        default="./data/test_countryroad.mp4", # TODO: make sure file format is mp4 and ...
        help="Path to the input video file"
    )

    parser.add_argument(
        "--num-levels",
        type=int,
        default=3,
        help="Number of pyramid levels to use (1 = no downsampling, 2+ = multi-scale)."
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Window size (odd number) for the Lucas-Kanade method."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory where output images/video are saved."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # 1. Load data
    frames = load_frames( args.input )

    processed_frames = []

    # 2. Pre-processing: build pyramids for each consecutive pair
    for i in range( len( frames ) - 1 ):
        frame1 = frames[ i ]
        frame2 = frames[ i+1 ]

        # Build Gaussian pyramids
        pyr1 = build_pyramid( frame1, num_levels=args.num_levels )
        pyr2 = build_pyramid( frame2, num_levels=args.num_levels )

        # Coarse level flow: Start from the top (smallest) level
        flow_coarse = lucas_kanade_flow( pyr1[ -1 ], pyr2[ -1 ], window_size=args.window_size )

        # (Optional) Upsample flow_coarse and refine at lower levels...
        # For now, let's just do single-level Lucas-Kanade on the largest pyramid level:
        flow_fine = lucas_kanade_flow( pyr1[ 0 ], pyr2[ 0 ], window_size=args.window_size )

        # 3. Visualization: overlay flow on original ( frame1 )
        flow_viz = visualize_flow( frame1, flow_fine, mode='color' ) # OR mode='arrows'
        
        # Save to PNG
        out_path = f"{args.output_dir}/flow_frame_{i}.png"
        save_output( flow_viz, f"output_flow_frame_{i}.png")

        # Add to processed_frames list for final video
        processed_frames.append( flow_viz )
    
    # Output to a single video
    video_out_path = f"{args.output_dir}/final_flow_video.mp4"
    write_video( processed_frames, video_out_path, fps=30.0 )

####################################
if __name__ == "__main__":
    main()