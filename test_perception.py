import cv2
import numpy as np
from perception import YOLOPerceptionEngine, BoundingBox, FrameData, LaneDetection

engine = YOLOPerceptionEngine(inference_every=1)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Test process_frame
try:
    fd = engine.process_frame(frame)
    print("process_frame succeeded")
except Exception as e:
    import traceback
    traceback.print_exc()

# Test visualise
try:
    vis = engine.visualise(frame, fd)
    print("visualise succeeded")
except Exception as e:
    import traceback
    traceback.print_exc()
