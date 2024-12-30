import cv2
import os


def load_frames( video_path: str ):
    """
    Reads a video file and returns a list of frames (as numpy arrays).

    Args:
        video_path (str): Path to the input video file.
    
    Returns:
        frames (List[ np.ndarray ] ): List of frames read from the video.
    """

    if not os.path.exists( video_path ):
        raise FileNotFoundError( f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture( video_path )
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append( frame )
    cap.release()

    if not frames:
        raise ValueError( f"No frames could be read from the video: {video_path}" )
    
    return frames

def save_output( image, output_path: str ):
    """
    Saves a single image (frame) to disk.

    Args:
        image (np.ndarray): The image to be saved
        output_path (str): Path (including filename) where the image will be saved.
    """
    # Create a directory, if the path does not exist
    os.makedirs( os.path.dirname( output_path), exist_ok=True )
    success = cv2.imwrite( output_path, image )

    if not success:
        raise IOError( f"Failed to write image to {output_path}" )

def write_video( frames, output_path: str, fps: float = 30.0 ):
    """
    Writes a list of frames (np.ndarray) to a video file.

    Args:
        frames ( List[ np.ndarray ] ): List of frames to be written.
        output_path (str): Path (including filename) of the output video.
        fps (float): Frames per second for the output video.
    """

    if not frames:
        raise ValueError( "No frames provided to write to video.")
    
    # Ensure output directory exists
    os.makedirs( os.path.dirname( output_path ), exist_ok=True )

    # Determine frame size from the first frame
    height, width, channels = frames[ 0 ].shape

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc( *'mp4v' )
    out = cv2.VideoWriter( output_path, fourcc, fps, (width, height) )

    for frame in frames:
        if frame.shape[0] != height or frame.shape[1] != width:
            raise ValueError( "All frames must have the same dimensions." )
        out.write( frame )

    out.release()