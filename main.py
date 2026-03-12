from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
import cv2

# ── Paths (update these) ──────────────────────────────────────────────
INPUT_VIDEO   = 'input_videos/input_video.mp4'
OUTPUT_VIDEO  = 'output_videos/output.avi'
BALL_MODEL    = 'models/yolo5_last.pt'
COURT_MODEL   = 'models/keypoints_model.pth'

def draw_bboxes(frame, player_dets, ball_dets):
    # Players
    for track_id, bbox in player_dets.items():
        x1,y1,x2,y2 = map(int, bbox)
        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,200,100), 2)
        cv2.putText(frame, f'P{track_id}', (x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,100), 2)
    # Ball
    for _, bbox in ball_dets.items():
        x1,y1,x2,y2 = map(int, bbox)
        cv2.ellipse(frame, (int((x1+x2)/2), int((y1+y2)/2)),
                    (10,10), 0, 0, 360, (0,255,255), 2)
    return frame

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

    print("✏️  Filtering to 2 players...")
    player_dets = player_tracker.choose_players(court_keypoints, player_dets)

    print("🎬 Drawing and saving output...")
    output_frames = []
    for i, frame in enumerate(frames):
        frame = draw_bboxes(frame, player_dets[i], ball_dets[i])
        # draw court keypoints
        for j in range(0, len(court_keypoints), 2):
            x, y = int(court_keypoints[j]), int(court_keypoints[j+1])
            cv2.circle(frame, (x,y), 5, (0,255,255), -1)
        output_frames.append(frame)

    save_video(output_frames, OUTPUT_VIDEO)
    print("✅ Done!")

if __name__ == '__main__':
    main()

