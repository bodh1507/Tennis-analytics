import cv2
import numpy as np
from utils.player_utils import get_foot_position, measure_distance


def pixel_to_real(pixel_point, H):
    """
    Convert pixel (x, y) to real-world coordinates (in meters)
    using the homography matrix H.
    Court coordinates are stored in cm (scale=100), so we divide by 100.
    Returns (x_m, y_m) tuple or None if H is None.
    """
    if H is None:
        return None
    pt = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]], dtype=np.float32)
    real = cv2.perspectiveTransform(pt, H)
    x_m = real[0][0][0] / 100.0   # cm → meters
    y_m = real[0][0][1] / 100.0
    return (x_m, y_m)


def compute_player_speeds(player_detections, H, fps):
    """
    Compute player speed in km/h for each frame.

    Strategy:
    - For each player, track their foot position across consecutive frames
    - Project foot positions to real-world meters using homography
    - speed = distance / time * 3.6 (m/s → km/h)
    - Cap at 40 km/h (realistic sprint speed for tennis)

    Returns:
        List of dicts per frame: [{player_id: {'speed_kmh': float}}, ...]
    """
    n = len(player_detections)
    speeds = [{} for _ in range(n)]

    # collect all unique player IDs
    player_ids = set()
    for det in player_detections:
        player_ids.update(det.keys())

    for pid in player_ids:
        # get all (frame_index, bbox) where this player is visible
        visible = [
            (i, det[pid])
            for i, det in enumerate(player_detections)
            if pid in det
        ]

        for idx in range(1, len(visible)):
            i_prev, bbox_prev = visible[idx - 1]
            i_curr, bbox_curr = visible[idx]

            foot_prev = get_foot_position(bbox_prev)
            foot_curr = get_foot_position(bbox_curr)

            real_prev = pixel_to_real(foot_prev, H)
            real_curr = pixel_to_real(foot_curr, H)

            if real_prev is None or real_curr is None:
                continue

            dist_m = measure_distance(real_prev, real_curr)
            frame_diff = i_curr - i_prev
            time_s = frame_diff / fps

            if time_s == 0:
                continue

            speed_kmh = (dist_m / time_s) * 3.6
            speed_kmh = min(speed_kmh, 40.0)   # cap at realistic sprint speed

            speeds[i_curr][pid] = {'speed_kmh': round(speed_kmh, 2)}

    return speeds
