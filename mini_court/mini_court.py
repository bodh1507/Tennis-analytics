import cv2
import numpy as np

# ── Standard tennis court dimensions (meters) ─────────────────────────
COURT_W = 10.97   # doubles width
COURT_H = 23.77   # full court length

# ── Mini court canvas size (pixels) ──────────────────────────────────
MINI_W  = 150
MINI_H  = 300
MARGIN  = 10
PAD     = 15

# Scale factors: pixels per meter on the mini court drawing
SCALE_X = (MINI_W - 2 * MARGIN) / COURT_W
SCALE_Y = (MINI_H - 2 * MARGIN) / COURT_H

# Player colors (alternating for P1 and P2)
PLAYER_COLORS = [(0, 200, 100), (255, 100, 0)]


def meters_to_mini(x_m, y_m):
    """Convert real-world meters to mini court pixel coordinates"""
    px = int(MARGIN + x_m * SCALE_X)
    py = int(MARGIN + y_m * SCALE_Y)
    # clamp to canvas bounds
    px = max(0, min(MINI_W - 1, px))
    py = max(0, min(MINI_H - 1, py))
    return (px, py)


def draw_mini_court(frame, player_positions_meters, ball_position_meters):
    """
    Draw a bird's-eye-view mini court in the bottom-right corner of the frame.

    Args:
        frame: current video frame (numpy array)
        player_positions_meters: {player_id: (x_m, y_m)}
        ball_position_meters: (x_m, y_m) or None

    Returns:
        frame with mini court overlay
    """
    h, w = frame.shape[:2]

    # ── Draw court canvas ─────────────────────────────────────────────
    mini = np.ones((MINI_H, MINI_W, 3), dtype=np.uint8) * 40  # dark bg

    # court surface (blue-ish like hard court)
    tl = meters_to_mini(0, 0)
    br = meters_to_mini(COURT_W, COURT_H)
    cv2.rectangle(mini, tl, br, (80, 60, 30), -1)   # filled court
    cv2.rectangle(mini, tl, br, (200, 200, 200), 1)  # outer boundary

    # net (thick white line at midpoint)
    net_l = meters_to_mini(0, COURT_H / 2)
    net_r = meters_to_mini(COURT_W, COURT_H / 2)
    cv2.line(mini, net_l, net_r, (255, 255, 255), 2)

    # service lines (top half)
    cv2.line(mini,
             meters_to_mini(0, COURT_H / 2 - 6.4),
             meters_to_mini(COURT_W, COURT_H / 2 - 6.4),
             (160, 160, 160), 1)

    # service lines (bottom half)
    cv2.line(mini,
             meters_to_mini(0, COURT_H / 2 + 6.4),
             meters_to_mini(COURT_W, COURT_H / 2 + 6.4),
             (160, 160, 160), 1)

    # center service line (vertical)
    cv2.line(mini,
             meters_to_mini(COURT_W / 2, COURT_H / 2 - 6.4),
             meters_to_mini(COURT_W / 2, COURT_H / 2 + 6.4),
             (160, 160, 160), 1)

    # singles sidelines
    cv2.line(mini, meters_to_mini(1.37, 0), meters_to_mini(1.37, COURT_H), (120, 120, 120), 1)
    cv2.line(mini, meters_to_mini(COURT_W - 1.37, 0), meters_to_mini(COURT_W - 1.37, COURT_H), (120, 120, 120), 1)

    # ── Draw players ─────────────────────────────────────────────────
    for i, (pid, (x_m, y_m)) in enumerate(player_positions_meters.items()):
        px, py = meters_to_mini(x_m, y_m)
        color = PLAYER_COLORS[i % 2]
        cv2.circle(mini, (px, py), 7, color, -1)
        cv2.putText(mini, f'P{pid}', (px + 6, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    # ── Draw ball ────────────────────────────────────────────────────
    if ball_position_meters is not None:
        bx, by = meters_to_mini(ball_position_meters[0], ball_position_meters[1])
        cv2.circle(mini, (bx, by), 4, (0, 255, 255), -1)

    # ── Paste mini court onto bottom-right of main frame ─────────────
    x_off = w - MINI_W - PAD
    y_off = PAD

    # draw a subtle border around the mini court
    cv2.rectangle(frame,
                  (x_off - 2, y_off - 2),
                  (x_off + MINI_W + 2, y_off + MINI_H + 2),
                  (200, 200, 200), 1)

    frame[y_off:y_off + MINI_H, x_off:x_off + MINI_W] = mini

    return frame
