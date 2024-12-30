import cv2
import numpy as np

def compute_gradients( frame1: np.ndarray, frame2: np.ndarray ):
    """
    Compute image gradients Ix, Iy for frame1 and the temporal gradient It between frame1 and frame2.

    This function returns:
        - Ix: partial derivative of frame1 wrt x
        - Iy: partial derivative of frame1 wrt y 
        - It: partial derivative wrt time, i.e., frame2 - frame1

    Args:
        frame1 (np.ndarray): Grayscale image at time t
        frame2 (np.ndarray): Grayscale image at time t+1

    Returns:
        Ix (np.ndarray): Gradient in x-direction
        Iy (np.ndarray): Gradient in y-direction
        It (np.ndarray): Gradient in t (frame2 - frame1)
    """

    # Ensure that inputs are grayscale
    if len( frame1.shape ) == 3:
        frame1 = cv2.cvtColor( frame1, cv2.COLOR_BGR2GRAY )
    
    if len( frame2.shape ) == 3:
        frame2 = cv2.cvtColor( frame2, cv2.COLOR_BGR2GRAY )
    
    # Convert to float32 for numerical stability
    frame1 = frame1.astype( np.float32 )
    frame2 = frame2.astype( np.float32 )

    # Compute spatial gradients (using Sobel as an example)
    Ix = cv2.Sobel( frame1, cv2.CV_32F, 1, 0, ksize = 3 )
    Iy = cv2.Sobel( frame1, cv2.CV_32F, 0, 1, ksize = 3 )

    # Temploral gradient ( simplistic: frame2 - frame1 )
    It = frame2 - frame1

    return Ix, Iy, It

def compute_spatial_gradients( frame: np.ndarray, ksize: int = 3, operator: str = 'sobel' ):
    """
    Compute spatial gradients (Ix, Iy) for a single frame using either Sobel or Scharr operators.

    Args:
        frame (np.ndarray): Grayscale image
        ksize (int): Kernel size for Sobel. Scharr doesn't use ksize in OpenCV (fixed internally).
        operator (str): 'sobel' or 'scharr' for different gradient operators

    Returns:
        Ix (np.ndarray), Iy (np.ndarray): spatial gradients in x and y directions
    """

    if len( frame.shape ) == 3:
        frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    
    frame = frame.astype( np.float32 )

    if operator.lower() == 'sobel':
        Ix = cv2.Sobel( frame, cv2.CV_32F, 1, 0, ksize=ksize )
        Iy = cv2.Sobel( frame, cv2.CV_32F, 0, 1, ksize=ksize )
    
    elif operator.lower() == 'scharr':
        # Scharr operator in OpenCV has superior rotation invariance but is not
        # parameterized by kernel size (the ksize is always 3 internally).
        Ix = cv2.Scharr( frame, cv2.CV_32F, 1, 0 )
        Iy = cv2.Scharr( frame, cv2.CV_32F, 0, 1 )
    else:
        raise ValueError( f"Unknown operator '{operator}'. Use 'sobel' or 'scharr'." )
    
    return Ix, Iy