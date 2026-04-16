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
                               draw_stats_table, draw_ball_trail,
                               draw_player_trails)
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


def get_ball_center(ball_dets):
    if not ball_dets:
        return None
    bbox = list(ball_dets.values())[0]
    return (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))


def point_to_bbox_distance(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float(np.sqrt(dx * dx + dy * dy))


def detect_hit_player(ball_center, player_boxes, frame_idx, last_hit_frame):
    if ball_center is None or frame_idx - last_hit_frame < 14:
        return None

    best_pid = None
    best_dist = float('inf')
    for pid, bbox in player_boxes.items():
        x1, y1, x2, y2 = bbox
        height = max(1, y2 - y1)
        hit_radius = max(55, height * 0.55)
        dist = point_to_bbox_distance(ball_center, bbox)
        if dist < best_dist and dist <= hit_radius:
            best_dist = dist
            best_pid = pid
    return best_pid


def detect_bounce(ball_history_px, ball_history_real, frame_idx, last_bounce_frame):
    if len(ball_history_px) < 3 or frame_idx - last_bounce_frame < 8:
        return None

    prev_pt, mid_pt, curr_pt = ball_history_px[-3:]
    dropped_into_bounce = mid_pt[1] - prev_pt[1] > 2
    rose_after_bounce = mid_pt[1] - curr_pt[1] > 2
    if dropped_into_bounce and rose_after_bounce:
        bounce_real = ball_history_real[-2]
        if bounce_real is None:
            return None
        x_m, y_m = bounce_real
        if -1.0 <= x_m <= 12.0 and -1.0 <= y_m <= 25.0:
            return bounce_real
    return None


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
    selected_player_ids = sorted({
        pid
        for frame_dets in player_dets
        for pid in frame_dets.keys()
    })[:2]
    running_player_stats = {
        pid: {
            'speed_kmh': 0.0,
            'shot_speed_kmh': 0.0,
            'avg_speed': 0.0,
            'avg_shot_speed': 0.0,
            'max_speed': 0.0,
            'max_shot_speed': 0.0,
            'distance_m': 0.0,
            'shot_count': 0,
        }
        for pid in selected_player_ids
    }

    # running averages for ball shot speed
    ball_speed_history = []
    ball_trail_px = []
    ball_history_real = []
    player_trails_px = {pid: [] for pid in selected_player_ids}
    previous_player_pos_m = {}
    total_distance_m = {pid: 0.0 for pid in selected_player_ids}
    shot_counts = {pid: 0 for pid in selected_player_ids}
    bounce_points_m = []
    last_hitter = None
    last_hit_frame = -999
    last_bounce_frame = -999

    print("🎬 Rendering output video...")
    output_frames = []
    for i, frame in enumerate(frames):
        # draw all elements
        frame = draw_frame_number(frame, i)
        frame = draw_court_keypoints(frame, court_keypoints)

        # ball speed
        ball_speed = get_ball_speed(ball_dets, i, H)
        if ball_speed is not None:
            ball_speed_history.append(ball_speed)

        ball_center_px = get_ball_center(ball_dets[i])
        ball_pos_m = None
        if ball_center_px is not None:
            ball_trail_px.append(ball_center_px)
            ball_pos_m = pixel_to_real(ball_center_px, H)
            ball_history_real.append(ball_pos_m)

        # build per-frame stats for table, carrying previous values so both
        # player columns stay visible on every frame.
        frame_stats = {
            pid: stats.copy()
            for pid, stats in running_player_stats.items()
        }
        current_speed_stats = player_speeds[i] if i < len(player_speeds) else {}
        for pid, stats in current_speed_stats.items():
            if pid in running_player_stats:
                running_player_stats[pid].update(stats)
                frame_stats[pid].update(stats)

        player_pos_m = {}
        for pid, bbox in player_dets[i].items():
            real = pixel_to_real(get_foot_position(bbox), H)
            if real:
                player_pos_m[pid] = real
                if pid in player_trails_px:
                    player_trails_px[pid].append(get_foot_position(bbox))
                if pid in previous_player_pos_m and pid in total_distance_m:
                    dist = measure_distance(previous_player_pos_m[pid], real)
                    if dist <= 1.2:
                        total_distance_m[pid] += dist
                previous_player_pos_m[pid] = real

        hit_player = detect_hit_player(
            ball_center_px, player_dets[i], i, last_hit_frame
        )
        if hit_player in shot_counts:
            shot_counts[hit_player] += 1
            last_hitter = hit_player
            last_hit_frame = i

        bounce_point = detect_bounce(
            ball_trail_px, ball_history_real, i, last_bounce_frame
        )
        if bounce_point is not None:
            bounce_points_m.append(bounce_point)
            last_bounce_frame = i

        # inject ball speed into stats
        avg_ball = (sum(ball_speed_history)/len(ball_speed_history)
                    if ball_speed_history else None)
        shot_speed = round(ball_speed, 1) if ball_speed else 0.0
        avg_shot_speed = round(avg_ball, 1) if avg_ball else 0.0
        for pid in selected_player_ids:
            if pid == last_hitter and ball_speed is not None:
                frame_stats[pid]['shot_speed_kmh'] = shot_speed
                frame_stats[pid]['max_shot_speed'] = max(
                    frame_stats[pid].get('max_shot_speed', 0.0),
                    shot_speed
                )
            frame_stats[pid]['avg_shot_speed'] = (
                avg_shot_speed
            )
            frame_stats[pid]['max_speed'] = max(
                frame_stats[pid].get('max_speed', 0.0),
                frame_stats[pid].get('speed_kmh', 0.0)
            )
            frame_stats[pid]['distance_m'] = round(total_distance_m[pid], 1)
            frame_stats[pid]['shot_count'] = shot_counts[pid]
            running_player_stats[pid].update(frame_stats[pid])

        frame = draw_player_trails(frame, player_trails_px)
        frame = draw_ball_trail(frame, ball_trail_px)
        frame = draw_player_bboxes(frame, player_dets[i])
        frame = draw_ball_bbox(frame, ball_dets[i])
        frame = draw_mini_court(
            frame, player_pos_m, ball_pos_m, bounce_points_m
        )
        frame = draw_stats_table(
            frame, frame_stats, ball_speed_history, i, selected_player_ids
        )
        output_frames.append(frame)

        if i % 50 == 0:
            print(f"   rendered {i}/{len(frames)} frames...")

    save_video(output_frames, OUTPUT_VIDEO)
    print(f"✅ Done! → {OUTPUT_VIDEO}")


if __name__ == '__main__':
    main()
