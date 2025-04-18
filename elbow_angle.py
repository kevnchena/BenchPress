import cv2
import mediapipe as mp
import numpy as np
import math

# 初始化 MediaPipe Pose 模組
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """計算三個點形成的夾角 (b 為中心)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def correct_projection_angle(proj_angle, horiz_angle, vert_angle):
    """修正因為 45° 斜角 + 俯視 角度造成的誤差"""
    # 轉換為弧度
    horiz_rad = np.radians(horiz_angle)
    vert_rad = np.radians(vert_angle)

    # 修正水平角度
    horiz_corrected = np.arctan(np.tan(np.radians(proj_angle)) * np.cos(horiz_rad))

    # 修正俯視角度
    real_angle = np.arctan(np.tan(horiz_corrected) * np.cos(vert_rad))

    return np.degrees(real_angle)

# 設定攝影機偏移角度
horiz_angle = 30  # 攝影機左右 45° 角度
vert_angle = 20   # 攝影機俯視 20°

# 讀取影片
video_name = "45_sample_03-21 000808.mp4"
viedo_angle = "Mediapipe_Output"
video_path = f"D://BenchPress_data//{viedo_angle}//{video_name}"  # 替換成你的影片名稱
cap = cv2.VideoCapture(video_path)

# 儲存修正後的角度變化
corrected_angles = []
frame_numbers = []

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換 BGR 影像為 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 偵測人體骨架
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # 取得關鍵點座標
        landmarks = results.pose_landmarks.landmark
        shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
        wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y])

        # 計算影像中的角度
        proj_angle = calculate_angle(shoulder, elbow, wrist)

        # 修正投影角度
        real_angle = correct_projection_angle(proj_angle, horiz_angle, vert_angle)

        corrected_angles.append(real_angle)
        frame_numbers.append(frame_index)

    frame_index += 1

cap.release()

# 繪製修正後的角度變化曲線
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, corrected_angles, marker='o', linestyle='-', color='g')
plt.xlabel("Frame")
plt.ylabel("degree")
plt.title(f"arm angle（horiz {horiz_angle}° + vert {vert_angle}°）")
plt.grid(True)
plt.show()

# 計算修正後的平均角度
if corrected_angles:
    avg_corrected_angle = np.mean(corrected_angles)
    print(f"average: {avg_corrected_angle:.2f}°")
else:
    print("cannot detect")