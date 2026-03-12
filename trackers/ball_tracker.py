from ultralytics import YOLO
import cv2
import pickle
import os
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # your fine-tuned YOLOv5/v8 ball model

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = [self.detect_frame(f) for f in frames]

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(detections, f)
        return detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        for box in results.boxes:
            return {1: box.xyxy.tolist()[0]}  # ball always class 1
        return {}

    def interpolate_ball_positions(self, detections):
        """Fill in missing ball frames using pandas interpolation"""
        from utils.player_utils import get_center_of_bbox

        positions = [get_center_of_bbox(d[1]) if 1 in d else (None, None)
                     for d in detections]

        df = pd.DataFrame(positions, columns=['x', 'y'])
        df = df.interpolate().bfill()  # linear interp + fill start gaps

        # convert back to bbox-style detections
        filled = []
        for _, row in df.iterrows():
            x, y = row['x'], row['y']
            filled.append({1: [x-5, y-5, x+5, y+5]})  # small bbox around center
        return filled
