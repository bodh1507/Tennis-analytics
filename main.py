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
                               draw_court_keypoints, draw_stats_panel,
                               draw_ball_speed)
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
from mini_court.mini_court import draw_mini_court
from analysis.player_stats import compute_player_speeds, pixel_to_real

# ── Paths ──────────────────────────────────────────────────────────────
INPUT_VIDEO  = 'input_videos/input_video.mp4'
OUTPUT_VIDEO = 'output_videos/output.avi'
BALL_MODEL   = 'models/yolo5_last.pt'
COURT_MODEL  = 'models/keypoints_model.pth'

# Real court keypoints in cm (scale=100 px/m)
COURT_PTS_REAL = np.array([
    [0,    0   ], [1097, 0   ],
    [0,    2377 ], [1097, 2377],
    [0,    1188 ], [1097, 1188],
    [200,  548  ], [897,  548 ],
    [200,  1829 ], [897,  1829],
    [548,  0    ], [548,  2377],
    [548,  548  ], [548,  1829],
], dtype=np.float32)


def compute_homography(court_keypoints_pixels):
    src = court_keypoints_pixels.reshape(14, 2).astype(np.float32)
    H, _ = cv2.findHomography(src, COURT_PTS_REAL)
    return H


def get_ball_speed(ball_dets, frame_idx, H, fps):
    if frame_idx == 0:
        return None
    prev = ball_dets[frame_idx-1]
    curr = ball_dets[frame_idx]
    if not prev or not curr:
        return None
    bbox_prev = list(prev.values())[0]
    bbox_curr = list(curr.values())[0]
    cx_p = (bbox_prev[0]+bbox_prev[2])/2
    cy_p = (bbox_prev[1]+bbox_prev[3])/2
    cx_c = (bbox_curr[0]+bbox_curr[2])/2
    cy_c = (bbox_curr[1]+bbox_curr[3])/2
    real_p = pixel_to_real((cx_p, cy_p), H)
    real_c = pixel_to_real((cx_c, cy_c), H)
    if real_p is None or real_c is None:
        return None
    dist = measure_distance(real_p, real_c)
    speed = dist / (1/fps) * 3.6
    return min(speed, 250.0)  # cap at 250 km/h


def main():
    print("📹 Reading video...")
    frames = read_video(INPUT_VIDEO)
    fps = 25
    print(f"   {len(frames)} frames")

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

    print("✏️  Filtering players...")
    player_dets = player_tracker.choose_players(court_keypoints, player_dets)

    print("📊 Computing player speeds...")
    player_speeds = compute_player_speeds(player_dets, H, fps)

    print("🎬 Rendering output...")
    output_frames = []
    for i, frame in enumerate(frames):
        # detections
        frame = draw_player_bboxes(frame, player_dets[i])
        frame = draw_ball_bbox(frame, ball_dets[i])
        frame = draw_court_keypoints(frame, court_keypoints)

        # ball speed
        ball_speed = get_ball_speed(ball_dets, i, H, fps)

        # player stats panel
        frame = draw_stats_panel(frame, player_speeds[i] if i < len(player_speeds) else {})
        frame = draw_ball_speed(frame, ball_speed)

        # mini court
        player_pos_m = {}
        for pid, bbox in player_dets[i].items():
            foot = get_foot_position(bbox)
            real = pixel_to_real(foot, H)
            if real:
                player_pos_m[pid] = real

        ball_pos_m = None
        if ball_dets[i]:
            bbox = list(ball_dets[i].values())[0]
            cx = (bbox[0]+bbox[2])/2
            cy = (bbox[1]+bbox[3])/2
            ball_pos_m = pixel_to_real((cx,cy), H)

        frame = draw_mini_court(frame, player_pos_m, ball_pos_m)

        output_frames.append(frame)
        if i % 50 == 0:
            print(f"   frame {i}/{len(frames)}")

    save_video(output_frames, OUTPUT_VIDEO)
    print("✅ Done! →", OUTPUT_VIDEO)


if __name__ == '__main__':
    main()