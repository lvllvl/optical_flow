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

    # Ensure frames are grayscale
    if len( frame1.shape ) == 3:
        frame1 = cv2.cvtColor( frame1, cv2.COLOR_BGR2GRAY )
    
    if len( frame2.shape ) == 3:
        frame2 = cv2.cvtColor( frame2, cv2.COLOR_BGR2GRAY )

    # Convert to float32 for better numeric stability
    frame1 = frame1.astype( np.float32 )
    frame2 = frame2.astype( np.float32 )

    # Calculate image gradients (Ix, Iy) and temporal gradient ( It )
        # options: custom gradient filters for more control ( e.g., Sobel, Scharr )
        # current: Using OpenCV's Sobel for spatial gradients and (frame2-frame1) for temporal gradient.
    Ix = cv2.Sobel( frame1, cv2.CV_32F, 1, 0, ksize=3 ) # partial derivative wrt x
    Iy = cv2.Sobel( frame1, cv2.CV_32F, 0, 1, ksize=3 ) # partial derivative wrt y
    It = frame2 - frame1                                # partial derivative wrt time 
    
    # Half window size
    half_w = window_size // 2

    # Prepare the flow array
    h, w = frame1.shape
    flow = np.zeros(( h, w, 2), dtype=np.float32 )

    # for each pixel, solve for ( u, v )
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