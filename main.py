from src.core.gradient import compute_gradients
from src.core.pyramid import build_pyramid
from src.core.flow_estimation import lucas_kanade_flow
from src.utils.io import load_frames, save_output, write_video
from src.utils.visualization import visualize_flow

import cv2
import numpy as np

def main( filename ):

    # 1. Load data
    frames = load_frames( filename )

    # 2. Pre-processing: build pyramids
    processed_frames = []
    for i in range( len( frames ) - 1 ):
        frame1 = frames[ i ]
        frame2 = frames[ i+1 ]

        # Build Gaussian pyramids
        pyr1 = build_pyramid( frame1, num_levels=3 )
        pyr2 = build_pyramid( frame2, num_levels=3 )

        # Start from the top (smallest) level
        flow_coarse = lucas_kanade_flow( pyr1[ -1 ], pyr2[ -1 ], window_size=7 )

        # (Optional) Upsample flow_coarse and refine at lower levels...
        # For now, let's just do single-level Lucas-Kanade on the largest pyramid level:
        flow_fine = lucas_kanade_flow( pyr1[ 0 ], pyr2[ 0 ], window_size=7 )

        # 3. Visualization: overlay flow on original ( frame1 )
        flow_viz = visualize_flow( frame1, flow_fine )
        out_path = f"outputs/flow_frame_{i}.png"
        save_output( flow_viz, f"output_flow_frame_{i}.png")
    
    # Output to a single video
    write_video( processed_frames, "outputs/final_flow_video.mp4", fps=30.0 )

####################################
if __name__ == "__main__":
    filename = './data/test_countryroad.mp4'
    main( filename )