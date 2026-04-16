import cv2
import numpy as np
from utils.player_utils import get_foot_position, measure_distance


def pixel_to_real(pixel_point, H):
    if H is None:
        return None
    pt = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]], dtype=np.float32)
    real = cv2.perspectiveTransform(pt, H)
    return (real[0][0][0] / 100.0, real[0][0][1] / 100.0)


def compute_player_speeds(player_detections, H, fps):
    """
    Returns list of per-frame stats dicts:
    [{player_id: {
        'speed_kmh': float,
        'shot_speed_kmh': None,     # placeholder, filled later
        'avg_speed': float,
        'avg_shot_speed': None
    }}, ...]
    """
    n = len(player_detections)
    speeds = [{} for _ in range(n)]

    player_ids = set()
    for det in player_detections:
        player_ids.update(det.keys())

    for pid in player_ids:
        visible = [
            (i, det[pid])
            for i, det in enumerate(player_detections)
            if pid in det
        ]

        all_speeds = []

        for idx in range(1, len(visible)):
            i_prev, bbox_prev = visible[idx-1]
            i_curr, bbox_curr = visible[idx]

            foot_prev = get_foot_position(bbox_prev)
            foot_curr = get_foot_position(bbox_curr)

            real_prev = pixel_to_real(foot_prev, H)
            real_curr = pixel_to_real(foot_curr, H)

            if real_prev is None or real_curr is None:
                continue

            dist_m = measure_distance(real_prev, real_curr)
            time_s = (i_curr - i_prev) / fps
            if time_s == 0:
                continue

            speed_kmh = min((dist_m / time_s) * 3.6, 40.0)
            all_speeds.append(speed_kmh)

            speeds[i_curr][pid] = {
                'speed_kmh':      round(speed_kmh, 1),
                'shot_speed_kmh': None,
                'avg_speed':      round(sum(all_speeds)/len(all_speeds), 1),
                'avg_shot_speed': None,
            }

    return speeds