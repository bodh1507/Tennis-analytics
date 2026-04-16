import cv2
import numpy as np


def draw_player_bboxes(frame, player_dets):
    """Draw red bounding boxes with Player ID labels"""
    for track_id, bbox in player_dets.items():
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f'Player ID: {track_id}'
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def draw_ball_bbox(frame, ball_dets):
    """Draw ball with label"""
    for bid, bbox in ball_dets.items():
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 255), 2)
        cv2.putText(frame, f'Ball ID: {bid}',
                    (cx - 30, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return frame


def draw_court_keypoints(frame, keypoints):
    """Draw numbered court keypoints"""
    for i in range(0, len(keypoints), 2):
        idx = i // 2
        x = int(keypoints[i])
        y = int(keypoints[i+1])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(idx), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return frame


def draw_frame_number(frame, frame_idx):
    """Draw frame counter top-left"""
    cv2.putText(frame, f'Frame: {frame_idx}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def draw_stats_table(frame, player_stats, ball_speeds, frame_idx, player_ids=None):
    """
    Draw a stats table in bottom-right like the reference output.
    Shows Shot Speed, Player Speed, Avg Shot Speed, Avg Player Speed
    for both players side by side.
    """
    h, w = frame.shape[:2]

    # Keep both table columns visible even when one player has no fresh speed
    # update on the current frame.
    pids = list(player_ids) if player_ids else sorted(player_stats.keys()) if player_stats else []
    while len(pids) < 2:
        pids.append(None)
    pids = pids[:2]

    # table dimensions and position: below the top-right mini court.
    tw, th = 230, 130
    mini_h = 300
    pad = 15
    gap = 25
    tx = w - tw - pad
    ty = pad + mini_h + gap
    if ty + th > h - 20:
        ty = h - th - 20

    # semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # header
    cv2.putText(frame, 'Player 1', (tx+80, ty+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
    cv2.putText(frame, 'Player 2', (tx+155, ty+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    # divider line
    cv2.line(frame, (tx+5, ty+30), (tx+tw-5, ty+30), (100,100,100), 1)

    rows = [
        ('Shot Speed',    'shot_speed_kmh'),
        ('Player Speed',  'speed_kmh'),
        ('avg. S. Speed', 'avg_shot_speed'),
        ('avg. P. Speed', 'avg_speed'),
    ]

    for r_idx, (label, key) in enumerate(rows):
        y = ty + 50 + r_idx * 22
        cv2.putText(frame, label, (tx+5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

        for p_idx, pid in enumerate(pids):
            x = tx + 78 + p_idx * 75
            if pid is not None and pid in player_stats and key in player_stats[pid]:
                val = player_stats[pid][key]
                if val is not None:
                    txt = f'{val:.1f} km/h'
                else:
                    txt = '0.0 km/h'
            else:
                txt = '0.0 km/h'
            color = (255, 255, 255)
            cv2.putText(frame, txt, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return frame
