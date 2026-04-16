import cv2
import numpy as np


def draw_ball_trail(frame, ball_trail, max_points=18):
    """Draw a fading trail for recent ball positions."""
    recent = ball_trail[-max_points:]
    for i in range(1, len(recent)):
        alpha = i / max(1, len(recent) - 1)
        color = (0, int(180 + 75 * alpha), 255)
        thickness = max(1, int(1 + 3 * alpha))
        cv2.line(frame, recent[i - 1], recent[i], color, thickness)
    for i, point in enumerate(recent):
        alpha = (i + 1) / max(1, len(recent))
        cv2.circle(frame, point, max(2, int(5 * alpha)), (0, 255, 255), -1)
    return frame


def draw_player_trails(frame, player_trails, max_points=24):
    """Draw short movement trails for each selected player."""
    colors = [(0, 200, 100), (255, 100, 0)]
    for idx, (pid, trail) in enumerate(sorted(player_trails.items())):
        recent = trail[-max_points:]
        color = colors[idx % len(colors)]
        for i in range(1, len(recent)):
            alpha = i / max(1, len(recent) - 1)
            faded = tuple(int(c * alpha + 40 * (1 - alpha)) for c in color)
            cv2.line(frame, recent[i - 1], recent[i], faded, 2)
    return frame


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
    tw, th = 275, 190
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
    cv2.putText(frame, 'Player 1', (tx+95, ty+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
    cv2.putText(frame, 'Player 2', (tx+185, ty+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    # divider line
    cv2.line(frame, (tx+5, ty+30), (tx+tw-5, ty+30), (100,100,100), 1)

    rows = [
        ('Shot Speed',   'shot_speed_kmh', 'km/h'),
        ('Max Shot',     'max_shot_speed', 'km/h'),
        ('Player Speed', 'speed_kmh', 'km/h'),
        ('Max Player',   'max_speed', 'km/h'),
        ('Avg Player',   'avg_speed', 'km/h'),
        ('Distance',     'distance_m', 'm'),
        ('Shots',        'shot_count', ''),
    ]

    for r_idx, (label, key, unit) in enumerate(rows):
        y = ty + 48 + r_idx * 20
        cv2.putText(frame, label, (tx+5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

        for p_idx, pid in enumerate(pids):
            x = tx + 95 + p_idx * 90
            if pid is not None and pid in player_stats and key in player_stats[pid]:
                val = player_stats[pid][key]
                if val is not None:
                    if key == 'shot_count':
                        txt = str(int(val))
                    elif unit:
                        txt = f'{val:.1f} {unit}'
                    else:
                        txt = f'{val:.1f}'
                else:
                    txt = f'0.0 {unit}'.strip()
            else:
                txt = f'0.0 {unit}'.strip()
            color = (255, 255, 255)
            cv2.putText(frame, txt, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)

    return frame
