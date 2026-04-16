import pickle
import os
from ultralytics import YOLO
from utils.player_utils import get_center_of_bbox, measure_distance


class PlayerTracker:
    def __init__(self, model_path='yolov8x.pt'):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
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

        return player_detections

    def detect_frame(self, frame):
        # conf=0.2 lower threshold catches the far/small player
        results = self.model.track(frame, persist=True, conf=0.2)[0]
        players = {}
        if results.boxes is None:
            return players
        for box in results.boxes:
            if box.id is None:
                continue
            # only keep 'person' class (class 0 in COCO)
            if int(box.cls[0]) != 0:
                continue
            track_id = int(box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]
            players[track_id] = bbox
        return players

    def choose_players(self, court_keypoints, player_detections):
        """
        Pick the 2 players closest to court across ALL frames,
        not just the first frame — fixes single player detection.
        """
        court_pts = [
            (int(court_keypoints[i]), int(court_keypoints[i+1]))
            for i in range(0, len(court_keypoints), 2)
        ]

        # accumulate distance scores across multiple frames
        player_scores = {}
        # check first 10 frames to be robust
        for frame_dict in player_detections[:10]:
            for track_id, bbox in frame_dict.items():
                center = get_center_of_bbox(bbox)
                dist = min(measure_distance(center, kp) for kp in court_pts)
                if track_id not in player_scores:
                    player_scores[track_id] = []
                player_scores[track_id].append(dist)

        # average distance per player
        avg_scores = {
            tid: sum(dists) / len(dists)
            for tid, dists in player_scores.items()
        }

        # pick 2 with lowest average distance to court
        sorted_ids = sorted(avg_scores, key=avg_scores.get)
        chosen_ids = set(sorted_ids[:2])
        print(f"  Chosen player IDs: {chosen_ids}")

        # filter all frames
        filtered = []
        for frame_dict in player_detections:
            filtered.append({
                tid: bbox
                for tid, bbox in frame_dict.items()
                if tid in chosen_ids
            })
        return filtered