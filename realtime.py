import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.8)

# 讀取影片檔案（這裡換成你自己的影片路徑）
video_path = 'D:\BenchPress_data\youtube//Bench Pressing 225lbs for 2 @115lbs bodyweight (1080p).mp4'  # 例如 '200kg_bench.mp4'
cap = cv2.VideoCapture(video_path)

# 記錄推舉軌跡
trajectory_points = []

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # 軌跡記錄 + 繪製
        wrist_pt = (int(wrist[0]), int(wrist[1]))
        trajectory_points.append(wrist_pt)
        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)

        # 畫點、線、角度
        cv2.circle(frame, tuple(map(int, shoulder)), 6, (0, 255, 0), -1)
        cv2.circle(frame, tuple(map(int, elbow)), 6, (0, 255, 0), -1)
        cv2.circle(frame, wrist_pt, 6, (0, 255, 0), -1)
        cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, elbow)), (255, 255, 255), 2)
        cv2.line(frame, tuple(map(int, elbow)), wrist_pt, (255, 255, 255), 2)

        # 顯示角度文字
        cv2.putText(frame, f"Elbow: {int(elbow_angle)} deg", tuple(map(int, elbow)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Bench Press Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()