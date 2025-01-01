# code to overlay flow vectors or create color coded flow maps
import cv2
import numpy as np

def visualize_flow( frame: np.ndarray, flow: np.ndarray, mode: str = 'color', step: int = 16 ) -> np.ndarray:
    """
    Visualize optical flow on the given frame.

    Args:
        frame (np.ndarray): Orignal frame ( BGR ).
        flow (np.ndarray): Flow field of shape ( H, W, 2 ), where flow[ ..., 0 ] = u (x), flow[ ..., 1 ] = v (y).
        mode (str): 'color' for color-coded flow, 'arrows' for arrow overlay.
        step (int): Step size for sampling flow vectors if using 'arrows'.
    
    Returns:
        vis (np.ndarray): A BGR image with the flow visualization.
    """
    if mode == 'color':
        return color_encode_flow( flow )
    elif mode == 'arrows':
        return draw_flow_arrows( frame, flow, step= step )
    else:
        raise ValueError( f"Unknown visualization mode: {mode}")
    
def color_encode_flow( flow: np.ndarray ) -> np.ndarray:
    """
    Color encode the flow field using an HSV-based approach.

    Args:
        flow (np.ndarray): (H,W,2) array of optical flow vectors (u,v).
    
    Returns:
        color_flow (np.ndarray): BGR image visualizing flow direction & magnitude.
    """

    # Separate flow into horizontal and vertical components
    u = flow[ ..., 0 ]
    v = flow[ ..., 1 ]

    # Convert flow to polar coordinates
    magnitude, angle = cv2.cartToPolar( u, v, angleInDegrees=True )

    # Create HSV image:
        # Hue = angle,
        # Saturation = 255,
        # Value = normalized magnitude
    hsv = np.zeros(( flow.shape[0], flow.shape[1], 3 ), dtype=np.uint8 )

    # Hue: angle / 2 if you want [ 0, 180 ] range. But since we used angleInDegrees=True
    # angles already go [ 0..360 ]. We will just clamp to [ 0..179 ] if needed.
    hsv[ ..., 0 ] = np.clip( angle / 2, 0, 179 ).astype( np.uint8 )
    hsv[ ..., 1 ] = 255
    hsv[ ..., 2 ] = cv2.normalize( magnitude, None, 0, 255, cv2.NORM_MINMAX )

    # Convert HSV to BGR for visualization
    color_flow = cv2.cvtColor( hsv, cv2.COLOR_HSV2BGR )

    return color_flow

def draw_flow_arrows( frame: np.ndarray, flow: np.ndarray, step: int = 16 ) -> np.ndarray:
    """
    Draw arrowed lines to visualize flow vectors on top of the original frame.

    Args:
        frame (np.ndarray): Original BGR frame.
        flow (np.ndarray): (H, W, 2) array of optical flow vectors (u, v).
        step (int): Step size for sampling vectors.

    Returns:
        vis (np.ndarray): BGR image with arrows drawn.
    """

    # Make a copy so we don't modify the original
    vis = frame.copy()
    h, w = vis.shape[ :2 ]

    for y in range( 0, h, step ):
        for x in range( 0, w, step ):

            fx, fy = flow[ y,x ] # flow vector at ( y,x )

            # End coordinates of arrow
            end_x = int( x+fx )
            end_y = int( y+fy )

            # Draw arrow
            cv2.arrowedLine(
                vis,
                ( x,y ),
                ( end_x, end_y ),
                color=( 0,255,0 ),
                thickness=1,
                tipLength=0.3
            )
    
    return vis