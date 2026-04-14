import pickle
import os
from ultralytics import YOLO
from utils.player_utils import get_center_of_bbox, measure_distance


class PlayerTracker:
    def __init__(self, model_path='yolov8x.pt'):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect and track players across all frames.
        Returns list of dicts: [{track_id: [x1,y1,x2,y2]}, ...]
        Uses stub (pickle cache) to avoid re-running YOLO every time.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"  Loading player detections from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        player_detections = []
        for frame in frames:
            players = self.detect_frame(frame)
            player_detections.append(players)

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) if os.path.dirname(stub_path) else '.', exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
            print(f"  Player detections saved to stub: {stub_path}")

        return player_detections

    def detect_frame(self, frame):
        """
        Run YOLO tracking on a single frame.
        Returns {track_id: [x1, y1, x2, y2]}
        """
        results = self.model.track(frame, persist=True, conf=0.3)[0]
        players = {}

        if results.boxes is None:
            return players

        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]
            players[track_id] = bbox

        return players

    def choose_players(self, court_keypoints, player_detections):
        """
        Filter detections to keep only the 2 actual players.
        Removes ball kids, referees, spectators by picking the 2
        people closest to the court keypoints in the first frame.
        """
        first_frame = player_detections[0]
        court_pts = [(int(court_keypoints[i]), int(court_keypoints[i+1]))
                     for i in range(0, len(court_keypoints), 2)]

        # find minimum distance from each person to any court keypoint
        min_dists = {}
        for track_id, bbox in first_frame.items():
            center = get_center_of_bbox(bbox)
            dist = min(measure_distance(center, kp) for kp in court_pts)
            min_dists[track_id] = dist

        # pick the 2 closest
        sorted_ids = sorted(min_dists, key=min_dists.get)
        chosen_ids = set(sorted_ids[:2])

        # filter all frames
        filtered = []
        for frame_dict in player_detections:
            filtered.append({
                tid: bbox
                for tid, bbox in frame_dict.items()
                if tid in chosen_ids
            })

        return filtered
