import cv2
import mediapipe as mp
import math

from fontTools.merge.util import first

# 初始化 Pose
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

# ---- 影片讀取 ----
video_name = "45_sample_03-21 000808.mp4"
viedo_angle = "45degree"
video_path = f"D://BenchPress_data//{viedo_angle}//{video_name}"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# ---- 參數初始化 ----
cam_angle = "L"
frame_idx = 0
rep_count = 0
rep_data_list = []

phase = "top"
region = "top"

top_range = 15
bottom_range = 30
top_threshold = None
bottom_threshold = None
first_rep = False

#補償threshold中失去的時間
if abs(fps - 30) <= 2:
    THRESHOLD_FRAME_PADDING = 2
elif abs(fps - 60) <= 2:
    THRESHOLD_FRAME_PADDING = 3
else:
    THRESHOLD_FRAME_PADDING = 2
threshold_time_padding = THRESHOLD_FRAME_PADDING / fps

wrist_highest = None
wrist_lowest = None

eccentric_start = 0
eccentric_end = 0
concentric_start = 0
concentric_end = 0

y_high = 0
y_low = 0

# ---- Mediapipe Pose ----
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def draw_line(frame, lmk1, lmk2, color=(200, 200, 200), thickness=3):
    cv2.line(frame, lmk1, lmk2, color, thickness)

def draw_point(frame, lmk, color=(255,255,255), size=4):
    cv2.circle(frame, lmk, size, color, -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Pose estimation
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    h, w, _ = frame.shape

    if results.pose_landmarks:
        # 先畫出 Mediapipe Pose 骨架
        landmarks = results.pose_landmarks.landmark
        # 走訪每個連結來畫線
        for con in mp_pose.POSE_CONNECTIONS:
            id1, id2 = con
            p1 = landmarks[id1]
            p2 = landmarks[id2]
            x1, y1 = int(p1.x*w), int(p1.y*h)
            x2, y2 = int(p2.x*w), int(p2.y*h)
            draw_point(frame, (x1, y1))
            draw_point(frame, (x2, y2))
            draw_line(frame, (x1, y1), (x2, y2), (200, 200, 200))

        # 抓取手腕
        WR_idx = 15 if cam_angle == "L" else 16
        WR = landmarks[WR_idx]
        WRy = int(WR.y * h)
        WRx = int(WR.x * w)

        # 初始化最高最低點
        if wrist_highest is None or WRy < wrist_highest[1]:
            wrist_highest = (WRx, WRy)
            top_threshold = WRy + top_range
#-----------針對第一下的底部偵測隔離------------------------
    #    if first_rep == False and wrist_lowest == None:
    #        wrist_lowest = (WRx, WRy+40)
    #        bottom_threshold = WRy - bottom_range
    #        first_gap = frame_idx+8
    #        print(f"frist = {first_gap}, idx = {frame_idx}")
#
    #    if frame_idx >= first_gap and first_rep == False:
    #        print("I'm here")
    #        wrist_lowest = (WRx, WRy)
    #        bottom_threshold = WRy - bottom_range
    #        first_rep = True
        if wrist_lowest is None or WRy > wrist_lowest[1]:
            wrist_lowest = (WRx, WRy)
            bottom_threshold = WRy - bottom_range

        # 畫出最高、最低點
        if wrist_highest:
            cv2.circle(frame, wrist_highest, 6, (0, 255, 255), -1)
            cv2.putText(frame, "Highest", (wrist_highest[0]+10, wrist_highest[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if wrist_lowest:
            cv2.circle(frame, wrist_lowest, 6, (0, 0, 255), -1)
            cv2.putText(frame, "Lowest", (wrist_lowest[0]+10, wrist_lowest[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 區域判定
        region = "middle"
        if WRy <= top_threshold :
            region = "top"
        elif WRy >= bottom_threshold:
            region = "bottom"
        if first_rep == False:
            region = "middle"
            first_rep = True
            print(region,phase)

        # 四狀態控制
        if phase == "top" and region == "middle":
            phase = "eccentric"
            eccentric_start = frame_idx
            print("離心開始")
            y_high = WRy

        elif phase == "eccentric" and region == "bottom":
            phase = "bottom"
            eccentric_end = frame_idx
            y_low = WRy
            print("離心結束")

        elif phase == "bottom" and region == "middle":
            phase = "concentric"
            concentric_start = frame_idx
            print("向心開始")

        elif phase == "concentric" and region == "top":
            phase = "top"
            concentric_end = frame_idx
            eccentric_time = (eccentric_end - eccentric_start)/fps + threshold_time_padding
            concentric_time = (concentric_end - concentric_start)/fps + threshold_time_padding
            print("向心結束")

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

        # 在畫面上顯示
        cv2.putText(frame, f"{region} ({phase})", (40,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    out.write(frame)
    cv2.imshow("BP with skeleton", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("======== process done ========")
print(f"總 reps: {rep_count}")
print("rep detail:")
for rep_obj in rep_data_list:
    print(rep_obj)