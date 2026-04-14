import cv2
import numpy as np


def draw_player_bboxes(frame, player_dets):
    """Draw bounding boxes and player IDs on frame"""
    for track_id, bbox in player_dets.items():
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
        cv2.putText(frame, f'P{track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)
    return frame


def draw_ball_bbox(frame, ball_dets):
    """Draw circle around detected ball"""
    for _, bbox in ball_dets.items():
        x1, y1, x2, y2 = map(int, bbox)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
    return frame


def draw_court_keypoints(frame, keypoints):
    """Draw court keypoints as yellow dots"""
    for i in range(0, len(keypoints), 2):
        x = int(keypoints[i])
        y = int(keypoints[i + 1])
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
    return frame


def draw_stats_panel(frame, player_stats):
    """
    Draw semi-transparent stats panel in top-left corner.
    player_stats: {player_id: {'speed_kmh': float}}
    """
    if not player_stats:
        return frame

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (290, 30 + 30 * len(player_stats)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = 35
    for player_id, stats in player_stats.items():
        spd = stats.get('speed_kmh', 0) or 0
        txt = f"P{player_id} Speed: {spd:.1f} km/h"
        cv2.putText(frame, txt, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 30

    return frame


def draw_ball_speed(frame, speed_kmh):
    """Draw ball speed below player stats panel"""
    if speed_kmh is None:
        return frame

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 110), (290, 148), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Ball Speed: {speed_kmh:.1f} km/h",
                (15, 138), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    return frame
