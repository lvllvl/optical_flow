# code for building Gaussian pyramids ( down / upsampling )
import cv2
import numpy as np

def build_pyramid(
        frame: np.ndarray,
        num_levels: int = 3,
        scale_factor: float = 0.5
) -> list:
    """
    Build a Gaussian pyramid for the given frame.

    Args:
        frame (np.ndarray): Input image (grayscale or color).
        num_levels (int): Number of levels in the pyramid.
                          1 = just the original image,
                          2 = original + 1 downsampled, etc.
        scale_factor (float): Downscall ratio between levels.

    Returns:
        pyramids (list): List of iamges 
    """
    pyramid = [ frame ]
    for _ in range( 1, num_levels ):
        
        # Use Gaussian blur then resize
        current_level = pyramid[ -1 ]

        # Optionally blur first: current level = cv2.GaussianBlur( current_level, (5,5), 1.0)
        next_level = cv2.resize(
            current_level,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR
        )

        pyramid.append( next_level )
    return pyramid