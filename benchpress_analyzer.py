import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np

from fontTools.merge.util import first

# 初始化 Pose
mp_pose = mp.solutions.pose

class BPPoint:
    def __init__(self,
                 y_high=0,
                 y_low=0,
                 depth_high=0.0,
                 depth_low=0.0,
                 eccentric_time=0.0,
                 concentric_time=0.0,
                 bottom_pause_time=0.0,
                 speed_unstable=False,
                 push_dips=False,
                 score=0.0,
                 rep=0):
        self.y_high = y_high
        self.y_low = y_low
        self.depth_high = depth_high
        self.depth_low = depth_low
        self.eccentric_time = eccentric_time
        self.concentric_time = concentric_time
        self.bottom_pause_time = bottom_pause_time
        self.speed_unstable = speed_unstable
        self.push_dips = push_dips
        self.score = score
        self.rep = rep

    def __repr__(self):
        return (f"BPPoint(y_high={self.y_high}, y_low={self.y_low}, "
                f" depth_high={self.depth_high:.2f}, depth_low={self.depth_low:.2f}, "
                f"eccentric_time={self.eccentric_time:.2f}, "
                f"concentric_time={self.concentric_time:.2f}, "
                f"bottom_pause_time={self.bottom_pause_time:.2f}, "
                f"push_dips={self.push_dips}, "
                f"speed_unstable={self.speed_unstable},"
                f"score={self.score}, rep={self.rep})")


def analyze_video(video_name: str, video_path: str, cam_angle: str, output_path: str, csv_path: str,
                  top_range=20, bottom_range=20, untable_threshold=2.5, dips_threshold=20):
    # ---- 影片讀取參數 ----
    video_name = video_name
    video_path = f"{video_path}//{video_name}"
    cam_angle = cam_angle

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    concentric_y_traj = []

    output_path = f"D://BenchPress_data//Mediapipe_Output//{video_name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"影片資訊:fps:{fps}, frame_width:{frame_width} ,frame_height:{frame_height}")

    # ---- Mediapipe Pose ----
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )

    #高低點門檻值
    top_range = top_range
    bottom_range = bottom_range
    #不穩定門檻值
    untable_threshold = untable_threshold
    #下陷門檻值
    dips_threshold = dips_threshold

# -----參數初始化 do not change-----
    frame_idx = 0
    rep_count = 0
    rep_data_list = []

    phase = "top"
    region = "top"

    top_threshold = None
    bottom_threshold = None
    first_rep = False
    concentric_y_traj = []
    concentric_dip_frames = []

    # 新增：紀錄深度變化
    wrist_shoulder_distance_high = 0.0
    wrist_shoulder_distance_low = 0.0


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

    #向心/離心、底部停留計時器
    eccentric_start = 0
    eccentric_end = 0
    concentric_start = 0
    concentric_end = 0
    bottom_pause_time = 0

    #高低點
    y_high = 0
    y_low = 0

# ------------------------------------

    #骨架繪製
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

            # 初始化最高
            if wrist_highest is None or WRy < wrist_highest[1]:
                wrist_highest = (WRx, WRy)
                top_threshold = WRy + top_range

            #初始化最低點，第一下以肩膀高度為準
            if not first_rep and wrist_lowest is None:
                shoulder_idx = 11 if cam_angle == "L" else 12
                shoulder_y = int(landmarks[shoulder_idx].y * h)
                wrist_lowest = (WRx, shoulder_y )  # 你可調整這個數值
                bottom_threshold = wrist_lowest[1] - bottom_range
                print(f"[Init] 使用肩膀高度模擬 wrist_lowest：{wrist_lowest}")

            #elif wrist_lowest is None or WRy > wrist_lowest[1]:
            #    wrist_lowest = (WRx, WRy)
            #    bottom_threshold = WRy - bottom_range

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

            # 四狀態控制
            if phase == "top" and region == "middle":
                phase = "eccentric"
                eccentric_start = frame_idx
                print("離心開始")
                y_high = WRy
                #紀錄肩膀-手腕的距離(high)
                shoulder_idx = 11 if cam_angle == "L" else 12
                SH = landmarks[shoulder_idx]
                shoulder_x, shoulder_y = int(SH.x * w), int(SH.y * h)
                wrist_shoulder_distance_high = ((WRx - shoulder_x) ** 2 + (WRy - shoulder_y) ** 2) ** 0.5

            elif phase == "eccentric" and region == "bottom":
                phase = "bottom"
                eccentric_end = frame_idx
                bottom_entry_frame = frame_idx
                y_low = WRy
                print("離心結束")

                # 紀錄肩膀-手腕的距離 (low)
                shoulder_idx = 11 if cam_angle == "L" else 12
                SH = landmarks[shoulder_idx]
                shoulder_x, shoulder_y = int(SH.x * w), int(SH.y * h)
                wrist_shoulder_distance_low = ((WRx - shoulder_x) ** 2 + (WRy - shoulder_y) ** 2) ** 0.5

            elif phase == "bottom" and region == "middle":
                phase = "concentric"
                concentric_start = frame_idx
                concentric_y_traj = []
                concentric_dip_frames = []  # 清空下陷紀錄
                #紀錄底部時間
                bottom_exit_frame = frame_idx
                bottom_pause_time = (bottom_exit_frame - bottom_entry_frame) / fps
                print("向心開始")

            elif phase == "concentric" and region == "top":
                phase = "top"
                concentric_end = frame_idx
                eccentric_time = (eccentric_end - eccentric_start)/fps# + threshold_time_padding
                concentric_time = (concentric_end - concentric_start)/fps #+ threshold_time_padding
                print("向心結束")

                # 判斷是否有回下陷（只要出現某一幀 WRy 又變大）
                push_dips = False
                for i in range(1, len(concentric_y_traj)):
                    if concentric_y_traj[i] > concentric_y_traj[i-1]+dips_threshold:
                        push_dips = True
                        concentric_dip_frames.append(WRy)
                        break

                #偵測向心速度穩定性
                velocities = [concentric_y_traj[i-1] - concentric_y_traj[i] for i in range(1, len(concentric_y_traj))]
                velocity_std = np.std(velocities)
                speed_unstable = velocity_std > untable_threshold  # 可依你影片的實際值調整門檻

                if eccentric_time >= 0.1 and concentric_time >= 0.1:
                    rep_count += 1
                    bp = BPPoint(
                        y_high=y_high,
                        y_low=y_low,
                        depth_high=wrist_shoulder_distance_high,
                        depth_low=wrist_shoulder_distance_low,
                        eccentric_time=eccentric_time,
                        concentric_time=concentric_time,
                        bottom_pause_time=bottom_pause_time,
                        speed_unstable=speed_unstable,
                        push_dips=push_dips,
                        rep=rep_count
                    )
                    rep_data_list.append(bp)
                    print(f"第{rep_count}次存入:{bp}")

                # 每幀記錄 WRy 並標出下陷點（影片中）
            if phase == "concentric" and WRy < y_low:
                concentric_y_traj.append(WRy)
                if len(concentric_y_traj) > 1 and WRy > concentric_y_traj[-2]:
                    concentric_dip_frames.append(WRy)

                    print("下陷偵測!")

                 # 在影片中畫出回下陷的點（紅色標記）並加入提示矩形區域
                for dip_y in concentric_dip_frames:
                    # 紅圈圈表示下陷點
                    cv2.circle(frame, (WRx, int(dip_y)), 6, (0, 0, 255), -1)
                    cv2.putText(frame, "Dip", (WRx + 10, int(dip_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # 顯示下陷區塊區域（紅色長方形）
                    cv2.rectangle(frame, (WRx - 20, int(dip_y) - 15), (WRx + 20, int(dip_y) + 15), (0, 0, 255), 1)
                    # 在畫面上顯示
            cv2.putText(frame, f"{region} ({phase})", (40,40),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

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

    # ✅ 匯出 CSV 給 Excel 分析
    rep_dict_list = [{
        "rep": r.rep,
        "y_high": r.y_high,
        "y_low": r.y_low,
        "depth_high": round(r.depth_high, 2),
        "depth_low": round(r.depth_low, 2),
        "eccentric_time": round(r.eccentric_time, 3),
        "concentric_time": round(r.concentric_time, 3),
        "bottom_pause_time": round(r.bottom_pause_time, 3),
        "speed_unstable": r.speed_unstable,
        "push_dips": r.push_dips,
        "score": r.score
    } for r in rep_data_list]

    df = pd.DataFrame(rep_dict_list)
    csv_path = f"D://BenchPress_data//output_data//{video_name}_benchpress_reps.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出到 CSV：{csv_path}")
    return csv_path, output_path
