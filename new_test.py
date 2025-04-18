import cv2
import mediapipe as mp
import math

# ---------------------【Pose 模組初始化】---------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------------------【顏色設定】---------------------
SHOULDER_TO_ELBOW_COLOR = (255, 0, 255)  # 紫色
ELBOW_TO_WRIST_COLOR = (255, 0, 0)  # 紅色
WRIST_TO_FINGER_COLOR = (255, 255, 255)  # 白色
HORIZONTAL_BODY_COLOR = (255, 255, 0)  # 黃色
VERTICAL_BODY_COLOR = (0, 255, 0)  # 綠色
LEG_COLOR = (255, 165, 0)  # 橙色
JOINT_COLOR = (255, 255, 255)  # 關節點顏色(白色)

# ---------------------【影片讀取與輸出】---------------------
video_name = "45_sample_03-21 000808.mp4"
viedo_angle = "45degree"
video_path = f"D://BenchPress_data//{viedo_angle}//{video_name}"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = f"D://BenchPress_data//Mediapipe_Output//{video_name}"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 用來儲存「按下 s 時」所紀錄的座標
start_points = {
    "L_shoulder": None,
    "R_shoulder": None,
    "L_wrist": None,
    "R_wrist": None,
    "L_ankle": None,
    "R_ankle": None
}


# ---------------------【計算三點角度函式(若需要)】---------------------
def angle_3pts(ax, ay, bx, by, cx, cy):
    """ 給定 A(ax, ay), B(bx, by), C(cx, cy)，以B為中心，計算A-B-C的夾角(0~180)。 """
    BAx, BAy = ax - bx, ay - by
    BCx, BCy = cx - bx, cy - by
    dot = (BAx * BCx + BAy * BCy)
    magBA = math.sqrt(BAx ** 2 + BAy ** 2)
    magBC = math.sqrt(BCx ** 2 + BCy ** 2)
    if magBA == 0 or magBC == 0:
        return 0.0
    cos_angle = dot / (magBA * magBC)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle


# ---------------------【主程式】---------------------
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 轉成 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape


            # (1) 繪製骨架 + 彩色連線
            def draw_line(landmark1, landmark2, color):
                point1 = (int(landmark1.x * w), int(landmark1.y * h))
                point2 = (int(landmark2.x * w), int(landmark2.y * h))
                cv2.line(frame, point1, point2, color, 3)


            def draw_point(landmark):
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                cv2.circle(frame, (px, py), 4, JOINT_COLOR, -1)


            # Mediapipe Pose Landmarks 參考:
            # 11: L_SHOULDER, 12: R_SHOULDER,
            # 15: L_WRIST,    16: R_WRIST,
            # 27: L_ANKLE,    28: R_ANKLE (以 BODY_25 或完整 POSE 模型為準)
            # 此處保留您原先指定方式
            for connection in mp_pose.POSE_CONNECTIONS:
                idx1, idx2 = connection
                lm1 = landmarks[idx1]
                lm2 = landmarks[idx2]

                # 繪製關節點
                draw_point(lm1)
                draw_point(lm2)

                # 顏色判斷(保持原本)
                if connection in [
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW)
                ]:
                    color = SHOULDER_TO_ELBOW_COLOR
                elif connection in [
                    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
                ]:
                    color = ELBOW_TO_WRIST_COLOR
                elif connection in [
                    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
                    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX)
                ]:
                    color = WRIST_TO_FINGER_COLOR
                elif connection in [
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                ]:
                    color = HORIZONTAL_BODY_COLOR
                elif connection in [
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP)
                ]:
                    color = VERTICAL_BODY_COLOR
                elif connection in [
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
                ]:
                    color = LEG_COLOR
                else:
                    color = (200, 200, 200)

                draw_line(lm1, lm2, color)

        # 顯示骨架與處理後畫面
        out.write(frame)
        cv2.imshow("Pose Detection", frame)

        # ---------------------【鍵盤偵測區】---------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 記錄座標 (若偵測到骨架)
            if results.pose_landmarks:
                # 取出 landmarks
                # (此處可能要做空值判斷)
                lm = results.pose_landmarks.landmark

                # shoulder(11,12), wrist(15,16), ankle(27,28)
                Lsh = (int(lm[11].x * w), int(lm[11].y * h))
                Rsh = (int(lm[12].x * w), int(lm[12].y * h))
                Lwr = (int(lm[15].x * w), int(lm[15].y * h))
                Rwr = (int(lm[16].x * w), int(lm[16].y * h))
                Lak = (int(lm[27].x * w), int(lm[27].y * h))
                Rak = (int(lm[28].x * w), int(lm[28].y * h))

                start_points["L_shoulder"] = Lsh
                start_points["R_shoulder"] = Rsh
                start_points["L_wrist"] = Lwr
                start_points["R_wrist"] = Rwr
                start_points["L_ankle"] = Lak
                start_points["R_ankle"] = Rak

                print("=== Start Points Saved ===")
                print(start_points)
                cv2.waitKey(0)
                if key == ord('s'):
                    continue

            else:
                print("No pose detected, cannot store start points.")
                cv2.waitKey()

        if key == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ 影片處理完成，已儲存為：{output_path}")
