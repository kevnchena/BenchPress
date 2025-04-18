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


# ---------------------【BPPoint 類別定義】---------------------
class BPPoint:
    """
    每個骨架關節資料
    x, y:            關節座標 (pixel)
    concentric_time: 向心階段經歷時間 (範例預設0.0)
    eccentric_time:  離心階段經歷時間 (範例預設0.0)
    push_dips:       推起時有無下沉 (True/False)
    score:           評分(可用於整體判斷, 範例預設0.0)
    """

    def __init__(self,
                 x=0, y=0,
                 concentric_time=0.0,
                 eccentric_time=0.0,
                 push_dips=False,
                 score=0.0):
        self.x = x
        self.y = y
        self.concentric_time = concentric_time
        self.eccentric_time = eccentric_time
        self.push_dips = push_dips
        self.score = score

    def __repr__(self):
        return (f"BPPoint(x={self.x}, y={self.y}, "
                f"concentric_time={self.concentric_time}, "
                f"eccentric_time={self.eccentric_time}, "
                f"push_dips={self.push_dips}, score={self.score})")


# ---------------------【初始化 points_dict】---------------------
points_dict = {
    "L_shoulder": BPPoint(),
    "R_shoulder": BPPoint(),
    "L_wrist": BPPoint(),
    "R_wrist": BPPoint(),
    "L_ankle": BPPoint(),
    "R_ankle": BPPoint()
}


# ---------------------【三點夾角函式(若需要)】---------------------
def angle_3pts(ax, ay, bx, by, cx, cy):
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

# ---------------------【欲觀察哪隻手腕】---------------------
# "L" = 左手腕, "R" = 右手腕
cam_angle = "L"

# ### 1) 新增：紀錄「全域最高點」與「全域最低點」(原程式功能) ###
wrist_highest = None
wrist_lowest = None

# ### 2) 新增：定義計次數與計時所需變數 ###
rep_count = 0  # 次數
phase = 'top'  # 初始假設在「頂部」(highest)
phase_start_frame = 0  # 進入某個 phase 的起始影格
time_list = []  # 紀錄每次完整動作(高->低->高)花費的時間(秒)

# ### 可自行微調: 若手腕y座標相對值< top_threshold 表示到達頂端附近,
#                 若> bottom_threshold 表示到底部附近 ###
#   下方可以用 "wrist_highest / wrist_lowest" 先行決定大致範圍
top_threshold = None
bottom_threshold = None

# Mediapipe Pose
pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
)


frame_idx = 0  # 用於計算時間( frame_idx / fps )
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1  # 影格計數+1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # 若偵測到姿勢
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape


        # 定義繪製的工具
        def draw_line(landmark1, landmark2, color):
            p1 = (int(landmark1.x * w), int(landmark1.y * h))
            p2 = (int(landmark2.x * w), int(landmark2.y * h))
            cv2.line(frame, p1, p2, color, 3)


        def draw_point(landmark):
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            cv2.circle(frame, (px, py), 4, JOINT_COLOR, -1)


        # 繪製骨架(簡化呈現)
        for connection in mp_pose.POSE_CONNECTIONS:
            idx1, idx2 = connection
            lm1 = landmarks[idx1]
            lm2 = landmarks[idx2]
            draw_point(lm1)
            draw_point(lm2)
            # 不細分顏色
            color = (200, 200, 200)
            draw_line(lm1, lm2, color)

        # 取「哪隻手腕」之 Landmark
        if cam_angle == "L":
            WR_idx = 15  # L_Wrist
        else:
            WR_idx = 16  # R_Wrist
        WR = landmarks[WR_idx]
        WRx = int(WR.x * w)
        WRy = int(WR.y * h)

        # 更新 points_dict
        if cam_angle == "L":
            points_dict["L_wrist"].x = WRx
            points_dict["L_wrist"].y = WRy
        else:
            points_dict["R_wrist"].x = WRx
            points_dict["R_wrist"].y = WRy

        # ---------------【原本 全域最高/最低點】---------------
        if wrist_highest is None:
            wrist_highest = (WRx, WRy)
            top_threshold = wrist_highest[1] + 10
        else:
            if WRy < wrist_highest[1]:  # 更上面( y更小 )
                wrist_highest = (WRx, WRy)

        if wrist_lowest is None:
            wrist_lowest = (WRx, WRy)
        else:
            if WRy > wrist_lowest[1]:  # 更下面( y更大 )
                wrist_lowest = (WRx, WRy)
                bottom_threshold = wrist_lowest[1] - 30

        # 在畫面顯示
        if wrist_highest:
            cv2.circle(frame, wrist_highest, 8, (0, 255, 255), -1)
            cv2.putText(frame, "Highest", (wrist_highest[0] + 10, wrist_highest[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if wrist_lowest:
            cv2.circle(frame, wrist_lowest, 8, (0, 0, 255), -1)
            cv2.putText(frame, "Lowest", (wrist_lowest[0] + 10, wrist_lowest[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


        # 做安全判斷
        region =""
        if top_threshold and bottom_threshold:
            # 區分 "top region" or "bottom region" or "middle"
            region = "middle"
            if WRy < top_threshold:  # 進入頂部區域
                region = "top"
            elif WRy > bottom_threshold:  # 進入底部區域
                region = "bottom"
            else:
                region = "middle"

            # 根據phase與region做狀態機
            # (top)->(bottom)->(top) 即記一次
            if phase == 'top':
                # 如果觀察到進入 bottom，則 phase切到bottom
                if region == 'bottom':
                    phase = 'bottom'
                    phase_start_frame = frame_idx  # 記錄一下開始時間(影格)
            elif phase == 'bottom':
                # 如果回到 top，就計1次
                if region == 'top':
                    phase = 'top'

    # --------------------------------------------------
    cv2.putText(frame,f"{region}",org = (50,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,thickness=2,color=(0,255,255))
    out.write(frame)
    cv2.imshow("Pose Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ 影片處理完成，已儲存為：{output_path}")
print("==================================================")
print(f"手腕最高點 (global) = {wrist_highest}")
print(f"手腕最低點 (global) = {wrist_lowest}")
print(f"--> rep_count : {rep_count}")
print("每次完整上->下->上所花時間(秒): ", time_list)
