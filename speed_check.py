import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

class BPPoint:
    def __init__(self,
                 y_high=0,
                 y_low=0,
                 eccentric_time=0.0,
                 concentric_time=0.0,
                 push_dips=False,
                 score=0.0,
                 rep=0):
        self.y_high = y_high
        self.y_low = y_low
        self.eccentric_time = eccentric_time
        self.concentric_time = concentric_time
        self.push_dips = push_dips
        self.score = score
        self.rep = rep

    def __repr__(self):
        return (f"BPPoint(y_high={self.y_high}, y_low={self.y_low}, "
                f"eccentric_time={self.eccentric_time:.2f}, "
                f"concentric_time={self.concentric_time:.2f}, "
                f"push_dips={self.push_dips}, "
                f"score={self.score}, rep={self.rep})")

video_path = "your_video.mp4"
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

top_range = 10
bottom_range = 30
top_threshold = None
bottom_threshold = None

wrist_highest = None
wrist_lowest = None

eccentric_start = 0
eccentric_end = 0
concentric_start = 0
concentric_end = 0
y_high = 0
y_low = 0

# 自動補幀數設定
if abs(fps - 30) <= 2:
    THRESHOLD_FRAME_PADDING = 2
elif abs(fps - 60) <= 2:
    THRESHOLD_FRAME_PADDING = 3
else:
    THRESHOLD_FRAME_PADDING = 2
threshold_time_padding = THRESHOLD_FRAME_PADDING / fps

last_WRy = None  # 速度計算用
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

while cap.isOpened():
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
        WR = landmarks[WR_idx]
        WRy = int(WR.y * h)

        # 計算速度
        if last_WRy is not None:
            delta_y = WRy - last_WRy
            velocity_y = delta_y / (1 / fps)
        else:
            velocity_y = 0
        last_WRy = WRy

        if wrist_highest is None or WRy < wrist_highest[1]:
            wrist_highest = (int(WR.x * w), WRy)
            top_threshold = WRy + top_range
        if wrist_lowest is None or WRy > wrist_lowest[1]:
            wrist_lowest = (int(WR.x * w), WRy)
            bottom_threshold = WRy - bottom_range

        region = "middle"
        if WRy < top_threshold:
            region = "top"
        elif WRy > bottom_threshold:
            region = "bottom"

        # ----------- 特殊處理第一下：用速度觸發離心 -----------
        if frame_idx < 30 and phase == "top" and abs(velocity_y) > 100:
            phase = "eccentric"
            eccentric_start = frame_idx
            y_high = WRy
            print("⚠️ 使用速度偵測第一下離心開始")

        # 四階段狀態控制
        if phase == "top" and region == "middle":
            phase = "eccentric"
            eccentric_start = frame_idx
            y_high = WRy

        elif phase == "eccentric" and region == "bottom":
            phase = "bottom"
            eccentric_end = frame_idx
            y_low = WRy

        elif phase == "bottom" and region == "middle":
            phase = "concentric"
            concentric_start = frame_idx

        elif phase == "concentric" and region == "top":
            phase = "top"
            concentric_end = frame_idx

            eccentric_time = (eccentric_end - eccentric_start) / fps + threshold_time_padding
            concentric_time = (concentric_end - concentric_start) / fps + threshold_time_padding

            if eccentric_time >= 0.1 and concentric_time >= 0.1:
                rep_count += 1
                bp = BPPoint(
                    y_high=y_high,
                    y_low=y_low,
                    eccentric_time=eccentric_time,
                    concentric_time=concentric_time,
                    rep=rep_count
                )
                rep_data_list.append(bp)
                print(f"[Rep {rep_count}] ecc={eccentric_time:.2f}s, con={concentric_time:.2f}s, yH={y_high}, yL={y_low}")

        # 繪圖：骨架、狀態、最高/最低點
        cv2.putText(frame, f"{region} ({phase})", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if wrist_highest:
            cv2.circle(frame, wrist_highest, 6, (0, 255, 255), -1)
            cv2.putText(frame, "Highest", (wrist_highest[0]+10, wrist_highest[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if wrist_lowest:
            cv2.circle(frame, wrist_lowest, 6, (0, 0, 255), -1)
            cv2.putText(frame, "Lowest", (wrist_lowest[0]+10, wrist_lowest[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("BP Rep Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("==============================")
print(f"✅ 總 reps: {rep_count}")
print("🔁 每次動作紀錄：")
for r in rep_data_list:
    print(r)
