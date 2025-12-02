import cv2
import os
import time

def webcam_on(userid, stop_flags_dict, seconds=60):

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("無法開啟攝影機")
        return None

    # 強制設定解析度（MP4V 最穩 720p）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("使用錄影解析度:", width, "x", height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    os.makedirs('../results', exist_ok=True)
    output_path = os.path.join('../results', f'{userid}.mp4')

    # 初始化 VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    print("VideoWriter opened:", out.isOpened())

    if not out.isOpened():
        print("MP4 Writer 初始化失敗，請改用 AVI (MJPG)")
        return None

    # 開始錄影
    start_time = time.time()
    while time.time() - start_time < seconds:

        # 偵測外部停止指令
        if stop_flags_dict.get(userid, False):
            print("偵測到停止指令")
            break

        ret, frame = cap.read()
        if not ret:
            print("⚠無法讀取畫面")
            break

        out.write(frame)

    # 收尾
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("webcam 錄影完成:", output_path)

    return output_path