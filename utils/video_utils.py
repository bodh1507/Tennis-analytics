import cv2


def read_video(path):
    """Read all frames from a video file into a list"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"  Read {len(frames)} frames from {path}")
    return frames


def save_video(frames, output_path, fps=24):
    """Write list of frames to an output video file"""
    if not frames:
        print("ERROR: no frames to save")
        return

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (w, h)
    )
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"  Video saved to {output_path}")
