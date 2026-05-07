import cv2

cap = cv2.VideoCapture("traffic_video.mp4")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", frame_count)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 2)
for i in range(5):
    ret, frame = cap.read()
    print(f"Read frame {frame_count - 2 + i}: {ret}")
    if not ret:
        print("End of video. Trying cap.set(1, 0)...")
        cap.set(1, 0)
        ret, frame = cap.read()
        print(f"Read after set(1, 0): {ret}")

cap.release()
