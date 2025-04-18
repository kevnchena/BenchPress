import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose 模組
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 顏色設定
SHOULDER_TO_ELBOW_COLOR = (255, 0, 255) # 紫色
ELBOW_TO_WRIST_COLOR = (255, 0, 0) # 紅色
WRIST_TO_FINGER_COLOR = (255, 255, 255) # 白色
HORIZONTAL_BODY_COLOR = (255, 255, 0) # 黃色
VERTICAL_BODY_COLOR = (0, 255, 0) # 綠色
LEG_COLOR = (255, 165, 0) # 橙色
JOINT_COLOR = (255, 255, 255) # 白色 (關節點)

# 讀取影片
video_name = "45_sample_03-21 000808.mp4"
viedo_angle = "45degree"
video_path = f"D://BenchPress_data//{viedo_angle}//{video_name}"  # 替換成你的影片名稱
cap = cv2.VideoCapture(video_path)

# 獲取影片基本資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定輸出影片
output_path = f"D://BenchPress_data//Mediapipe_Output//{video_name}"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 初始化 Pose 模型
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 影片結束

        # 轉換 BGR 影像為 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 偵測骨架
        results = pose.process(rgb_frame)

        # 如果有偵測到人體骨架
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # 繪製線條的函數
            def draw_line(landmark1, landmark2, color):
                point1 = (int(landmark1.x * w), int(landmark1.y * h))
                point2 = (int(landmark2.x * w), int(landmark2.y * h))
                cv2.line(frame, point1, point2, color, 3)

            # 繪製關節點
            def draw_point(landmark):
                point = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, point, 4, JOINT_COLOR, -1)

            # 繪製骨架線條並依照部位上色
            for connection in mp_pose.POSE_CONNECTIONS:
                landmark1 = landmarks[connection[0]]
                landmark2 = landmarks[connection[1]]

                # 繪製關節點
                draw_point(landmark1)
                draw_point(landmark2)

                # 判斷顏色
                if connection in [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                                   (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW)]:
                    color = SHOULDER_TO_ELBOW_COLOR
                elif connection in [(mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                                     (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)]:
                    color = ELBOW_TO_WRIST_COLOR
                elif connection in [(mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
                                     (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX)]:
                    color = WRIST_TO_FINGER_COLOR
                elif connection in [(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                                     (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)]:
                    color = HORIZONTAL_BODY_COLOR
                elif connection in [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                                     (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP)]:
                    color = VERTICAL_BODY_COLOR
                elif connection in [(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                                     (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                                     (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                                     (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)]:
                    color = LEG_COLOR
                else:
                    color = (200, 200, 200)  # 預設灰色

                draw_line(landmark1, landmark2, color)

        # 儲存新影片
        out.write(frame)

        # 顯示即時畫面 (可選擇關閉)
        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 按 'q' 退出
            break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ 影片處理完成，已儲存為：{output_path}")
