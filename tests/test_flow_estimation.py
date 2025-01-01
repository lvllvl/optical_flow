import pytest
import numpy as np
from src.core.flow_estimation import lucas_kanade_flow

def test_lucas_kande_zero_flow():

    # Create two identical frames
    frame = np.zeros( (100, 100), dtype=np.uint8 )
    flow = lucas_kanade_flow( frame1=frame, frame2=frame )

    # Expect zero flow everywhere
    assert np.allclose( flow, 0 )
