from ultralytics import YOLO
import cv2

class PlayerTracker:
    def __init__(self, model_path='yolov8x.pt'):
        self.model = YOLO(model_path)  # base YOLOv8 for players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Returns list of dicts: [{track_id: bbox}, ...] per frame"""
        import pickle, os

        # stub = cached detections so you don't re-run YOLO every time
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        player_detections = []
        for frame in frames:
            players = self.detect_frame(frame)
            player_detections.append(players)

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        players = {}
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]
            players[track_id] = bbox
        return players

    def choose_players(self, court_keypoints, player_detections):
        """Keep only the 2 players closest to the court — filters out ball kids, referees"""
        from utils.player_utils import get_center_of_bbox, measure_distance

        # use first frame to pick the 2 court players
        first = player_detections[0]
        chosen = []
        min_dists = {}

        for track_id, bbox in first.items():
            center = get_center_of_bbox(bbox)
            dist = min(
                measure_distance(center, (int(kp[0]), int(kp[1])))
                for kp in court_keypoints.reshape(-1, 2)
            )
            min_dists[track_id] = dist

        sorted_ids = sorted(min_dists, key=min_dists.get)
        chosen = sorted_ids[:2]

        # filter all frames to only keep chosen player IDs
        filtered = []
        for frame_dict in player_detections:
            filtered.append({tid: bbox for tid, bbox in frame_dict.items() if tid in chosen})
        return filtered
