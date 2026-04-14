import pickle
import os
import pandas as pd
from ultralytics import YOLO
from utils.player_utils import get_center_of_bbox


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect ball across all frames.
        Returns list of dicts: [{1: [x1,y1,x2,y2]}, ...] (ball always key=1)
        Uses stub cache to avoid re-running YOLO.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"  Loading ball detections from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = []
        for frame in frames:
            det = self.detect_frame(frame)
            detections.append(det)

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) if os.path.dirname(stub_path) else '.', exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(detections, f)
            print(f"  Ball detections saved to stub: {stub_path}")

        return detections

    def detect_frame(self, frame):
        """
        Run ball detection on a single frame.
        Low confidence threshold (0.15) because the ball is tiny.
        Returns {1: [x1, y1, x2, y2]} or {} if not found.
        """
        results = self.model.predict(frame, conf=0.15)[0]
        for box in results.boxes:
            return {1: box.xyxy.tolist()[0]}
        return {}

    def interpolate_ball_positions(self, ball_detections):
        """
        Fill in missing ball detections using pandas linear interpolation.
        The ball disappears in frames due to motion blur or occlusion.
        We estimate position by interpolating between known positions.
        """
        # extract center positions
        positions = []
        for det in ball_detections:
            if 1 in det:
                bbox = det[1]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                positions.append({'x': cx, 'y': cy})
            else:
                positions.append({'x': None, 'y': None})

        df = pd.DataFrame(positions)
        df = df.interpolate()   # linear interpolation
        df = df.bfill()         # fill any remaining NaN at the start

        # convert back to bbox-style detections
        filled = []
        for _, row in df.iterrows():
            x, y = row['x'], row['y']
            if pd.isna(x) or pd.isna(y):
                filled.append({})
            else:
                # small 10x10 box around interpolated center
                filled.append({1: [x - 5, y - 5, x + 5, y + 5]})

        return filled
