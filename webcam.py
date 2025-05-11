import cv2
import os
import mediapipe as mp
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils
#
# pose = mp_pose.Pose(
#    static_image_mode=False,
#    model_complexity=2,
#    smooth_landmarks=True,
#    enable_segmentation=False,
#    min_detection_confidence=0.8,
#    min_tracking_confidence=0.8
# )


def webcam_on(userid):
    # 開啟預設攝影機（通常是 webcam）
    cap = cv2.VideoCapture(0)

    # 檢查是否成功打開攝影機
    if not cap.isOpened():
        print("無法開啟攝影機")
        exit()

    # 設定影片格式與輸出檔案
    print("攝影機解析度:", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 視訊編碼格式
    # 檔名, 格式, FPS, 畫面尺寸
    output_path = os.path.join('results', f'{userid}.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
            break
    # ----mediapipe----
        # 轉換 BGR 到 RGB（MediaPipe 需要）
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
        # 使用 MediaPipe 偵測姿勢
        # results = pose.process(rgb)
    #
        # 如果偵測到骨架，畫出來
        # if results.pose_landmarks:
        #    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # ----mediapipe----
        # 寫入影片
        out.write(frame)

        # 顯示畫面
        cv2.imshow('Webcam', frame)

        # 按下 q 鍵離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return f".\{output_path}"
