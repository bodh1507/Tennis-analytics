import torch, functools
_real_load = torch.load
@functools.wraps(_real_load)
def _patched(*args, **kwargs):
    kwargs['weights_only'] = False
    return _real_load(*args, **kwargs)
torch.load = _patched

import cv2
import numpy as np
from utils.video_utils import read_video, save_video
from utils.player_utils import get_foot_position, measure_distance
from utils.draw_utils import (draw_player_bboxes, draw_ball_bbox,
                               draw_court_keypoints, draw_frame_number,
                               draw_stats_table)
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
from mini_court.mini_court import draw_mini_court
from analysis.player_stats import compute_player_speeds, pixel_to_real

INPUT_VIDEO  = 'input_videos/input_video.mp4'
OUTPUT_VIDEO = 'output_videos/output.avi'
BALL_MODEL   = 'models/yolo5_last.pt'
COURT_MODEL  = 'models/keypoints_model.pth'
FPS          = 25

COURT_PTS_REAL = np.array([
    [0,    0   ], [1097, 0   ],
    [0,    2377 ], [1097, 2377],
    [0,    1188 ], [1097, 1188],
    [200,  548  ], [897,  548 ],
    [200,  1829 ], [897,  1829],
    [548,  0    ], [548,  2377],
    [548,  548  ], [548,  1829],
], dtype=np.float32)


def compute_homography(kps):
    src = kps.reshape(14, 2).astype(np.float32)
    H, _ = cv2.findHomography(src, COURT_PTS_REAL)
    return H


def get_ball_speed(ball_dets, frame_idx, H):
    if frame_idx == 0 or not ball_dets[frame_idx] or not ball_dets[frame_idx-1]:
        return None
    def center(bbox): return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
    p = pixel_to_real(center(list(ball_dets[frame_idx-1].values())[0]), H)
    c = pixel_to_real(center(list(ball_dets[frame_idx].values())[0]), H)
    if p is None or c is None: return None
    return min(measure_distance(p, c) * FPS * 3.6, 250.0)


def main():
    print("📹 Reading video...")
    frames = read_video(INPUT_VIDEO)
    print(f"   {len(frames)} frames loaded")

    print("🔍 Detecting players...")
    player_tracker = PlayerTracker('yolov8x.pt')
    player_dets = player_tracker.detect_frames(
        frames, read_from_stub=True,
        stub_path='tracker_stubs/player_detections.pkl'
    )

    print("🎾 Detecting ball...")
    ball_tracker = BallTracker(BALL_MODEL)
    ball_dets = ball_tracker.detect_frames(
        frames, read_from_stub=True,
        stub_path='tracker_stubs/ball_detections.pkl'
    )
    ball_dets = ball_tracker.interpolate_ball_positions(ball_dets)

    print("🏟️  Detecting court keypoints...")
    court_detector = CourtLineDetector(COURT_MODEL)
    court_keypoints = court_detector.predict(frames[0])

    print("📐 Computing homography...")
    H = compute_homography(court_keypoints)

    print("✏️  Choosing 2 players...")
    player_dets = player_tracker.choose_players(court_keypoints, player_dets)

    print("📊 Computing player speeds...")
    player_speeds = compute_player_speeds(player_dets, H, FPS)

    # running averages for ball shot speed
    ball_speed_history = []

    print("🎬 Rendering output video...")
    output_frames = []
    for i, frame in enumerate(frames):
        # draw all elements
        frame = draw_frame_number(frame, i)
        frame = draw_player_bboxes(frame, player_dets[i])
        frame = draw_ball_bbox(frame, ball_dets[i])
        frame = draw_court_keypoints(frame, court_keypoints)

        # ball speed
        ball_speed = get_ball_speed(ball_dets, i, H)
        if ball_speed is not None:
            ball_speed_history.append(ball_speed)

        # build per-frame stats for table
        frame_stats = player_speeds[i] if i < len(player_speeds) else {}

        # inject ball speed into stats
        pids = list(frame_stats.keys())
        avg_ball = (sum(ball_speed_history)/len(ball_speed_history)
                    if ball_speed_history else None)
        for pid in pids:
            frame_stats[pid]['shot_speed_kmh'] = ball_speed
            frame_stats[pid]['avg_shot_speed'] = (
                round(avg_ball, 1) if avg_ball else None
            )

        # draw stats table
        frame = draw_stats_table(frame, frame_stats, ball_speed_history, i)

        # mini court
        player_pos_m = {}
        for pid, bbox in player_dets[i].items():
            real = pixel_to_real(get_foot_position(bbox), H)
            if real:
                player_pos_m[pid] = real

        ball_pos_m = None
        if ball_dets[i]:
            bbox = list(ball_dets[i].values())[0]
            ball_pos_m = pixel_to_real(
                ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2), H
            )

        frame = draw_mini_court(frame, player_pos_m, ball_pos_m)
        output_frames.append(frame)

        if i % 50 == 0:
            print(f"   rendered {i}/{len(frames)} frames...")

    save_video(output_frames, OUTPUT_VIDEO)
    print(f"✅ Done! → {OUTPUT_VIDEO}")


if __name__ == '__main__':
    main()