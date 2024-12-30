from src.core.gradient import compute_gradients
import numpy as np
import cv2

def lucas_kanade_flow( frame1: np.ndarray, frame2: np.ndarray, window_size: int=5 ) -> np.ndarray:
    """
    Compute the optical flow using the basic Lucas-Kanade method.

    Args:
        frame1 (np.ndarray): Grayscale image at time t
        frame2 (np.ndarray): Grayscale image at time t+1
        window_size (int): Size of local neighborhood (odd number). Typical default: 5 or 7.

    Returns:
        flow (np.ndarray): An (H, W, 2) array representing flow vectors (u, v) at each pixel.
                           flow[ y, x, 0 ] = horizontal flow (u),
                           flow[ y, x, 1 ] = vertical flow (v).

    """
    # 1. Convert frames to grayscale + compute Ix, Iy, It
    Ix, Iy, It = compute_gradients( frame1, frame2 )

    # Ensure they are fload32, if not already
    Ix = Ix.astype( np.float32 )
    Iy = Iy.astype( np.float32 )
    It = It.astype( np.float32 )
    
    # Prepare the flow array
    h, w = Ix.shape
    flow = np.zeros(( h, w, 2), dtype=np.float32 )

    # Half window size
    half_w = window_size // 2

    # 2. for each pixel, solve for ( u, v )
    for y in range( half_w, h-half_w ):
        for x in range( half_w, w-half_w ):

            # Local window indices
            y0, y1 = ( y - half_w ), ( y + half_w + 1 )
            x0, x1 = ( x - half_w ), ( x + half_w + 1 )

            # Extract local patches of derivatives
            Ix_patch = Ix[ y0:y1, x0:x1 ].flatten()
            Iy_patch = Iy[ y0:y1, x0:x1 ].flatten()
            It_patch = It[ y0:y1, x0:x1 ].flatten()

            # Construct A and b in "A * v = b", where v = [ u,v ]
            # A = [ [ Ix1, Iy1 ],
            #         Ix2, Iy2 ],
            #         ...
            #       [ IxN, IyN ] ]
            # b = [ -It1, -It2, ..., -ItN ]^T
            A = np.vstack( ( Ix_patch, Iy_patch )).T
            b = -It_patch

            # Solve for v = ( A^T * A )^{-1} * A^T * b
            # In practice, we might want to check for singularities
            # ( e.g., if A^T * A is non-invertible )
            ATA = A.T @ A
            if np.linalg.cond( ATA ) < 1e3: # arbitrary threshold, to skip unstable solutions
                v = np.linalg.inv( ATA )@ A.T @ b
                flow[ y, x, 0 ] = v[ 0 ] # u
                flow[ y, x, 1 ] = v[ 1 ] # v 

    return flow