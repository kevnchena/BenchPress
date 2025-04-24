# ✅ Bench Press 分析：四狀態 + 深度估計 + 推起下陷判斷 + 匯出 CSV

import cv2
import mediapipe as mp
import pandas as pd
import math

mp_pose = mp.solutions.pose

class BPPoint:
    def __init__(self, y_high=0, y_low=0, eccentric_time=0.0, concentric_time=0.0, depth_change=0.0, push_dips=False, score=0.0, rep=0):
        self.y_high = y_high
        self.y_low = y_low
        self.eccentric_time = eccentric_time
        self.concentric_time = concentric_time
        self.depth_change = depth_change  # ✅ 新增：深度變化量
        self.push_dips = push_dips
        self.score = score
        self.rep = rep

    def __repr__(self):
        return (f"BPPoint(y_high={self.y_high}, y_low={self.y_low}, "
                f"eccentric_time={self.eccentric_time:.2f}, "
                f"concentric_time={self.concentric_time:.2f}, "
                f"depth_change={self.depth_change:.2f}, "
                f"push_dips={self.push_dips}, rep={self.rep})")

video_name = "45_sample_03-21 000808.mp4"
viedo_angle = "45degree"
video_path = f"D://BenchPress_data//{viedo_angle}//{video_name}"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

cam_angle = "L"
frame_idx = 0
rep_count = 0
rep_data_list = []

phase = "top"
region = "top"
concentric_y_traj = []
concentric_depth_traj = []  # ✅ 新增：用來記錄向心階段的肩腕距離

top_range = 15
bottom_range = 30
top_threshold = None
bottom_threshold = None
first_rep = False

pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                    enable_segmentation=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    h, w, _ = frame.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        WR_idx = 15 if cam_angle == "L" else 16
        SH_idx = 11 if cam_angle == "L" else 12
        WRx, WRy = int(landmarks[WR_idx].x * w), int(landmarks[WR_idx].y * h)
        SHx, SHy = int(landmarks[SH_idx].x * w), int(landmarks[SH_idx].y * h)

        depth_distance = math.sqrt((WRx - SHx)**2 + (WRy - SHy)**2)

        # 畫點與骨架
    for connection in mp_pose.POSE_CONNECTIONS:
        id1, id2 = connection
        lm1 = landmarks[id1]
        lm2 = landmarks[id2]
        x1, y1 = int(lm1.x * w), int(lm1.y * h)
        x2, y2 = int(lm2.x * w), int(lm2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
        cv2.circle(frame, (x1, y1), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x2, y2), 4, (255, 255, 255), -1)

            # 區域判定與四階段狀態機
    region = "middle"
    if WRy <= top_threshold:
        region = "top"
    elif WRy >= bottom_threshold:
        region = "bottom"

    if phase == "top" and region == "middle":
        phase = "eccentric"
        eccentric_start = frame_idx
        y_high = WRy

    elif phase == "eccentric" and region == "bottom":
        phase = "bottom"
        eccentric_end = frame_idx
        y_low = WRy

    if phase == "bottom" and region == "middle":
        phase = "concentric"
        concentric_start = frame_idx
        concentric_y_traj = []
        concentric_depth_traj = []  # ✅ 開始記錄肩腕距離
    elif phase == "concentric" and region == "top":
        phase = "top"
        concentric_end = frame_idx
        eccentric_time = (eccentric_end - eccentric_start)/fps
        concentric_time = (concentric_end - concentric_start)/fps
        push_dips = any(concentric_y_traj[i] > concentric_y_traj[i-1] + 2 for i in range(1, len(concentric_y_traj)))
        depth_change = max(concentric_depth_traj) - min(concentric_depth_traj)  # ✅ 計算深度變化
        if eccentric_time >= 0.1 and concentric_time >= 0.1:
            rep_count += 1
            bp = BPPoint(y_high=y_high, y_low=y_low, eccentric_time=eccentric_time, concentric_time=concentric_time,
                         depth_change=depth_change, push_dips=push_dips, rep=rep_count)
            rep_data_list.append(bp)
    # 向心階段持續記錄
    if phase == "concentric":
        concentric_y_traj.append(WRy)
        concentric_depth_traj.append(depth_distance)

    out.write(frame)
    cv2.imshow("BP Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# 匯出 CSV
rep_dict_list = [{
    "rep": r.rep,
    "y_high": r.y_high,
    "y_low": r.y_low,
    "eccentric_time": round(r.eccentric_time, 3),
    "concentric_time": round(r.concentric_time, 3),
    "depth_change": round(r.depth_change, 2),
    "push_dips": r.push_dips
} for r in rep_data_list]

pd.DataFrame(rep_dict_list).to_csv(f"D://BenchPress_data//{video_name}_reps.csv", index=False, encoding="utf-8-sig")
