import cv2

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, path, fps=24):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps, (w, h)
    )
    for f in frames:
        out.write(f)
    out.release()
    print(f"✅ Video saved: {path}")
