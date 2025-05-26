import cv2
import os
import time

def webcam_on(userid, stop_flags_dict, seconds=60):
    # 開啟預設攝影機（通常是 webcam）
    cap = cv2.VideoCapture(0)

    # 檢查是否成功打開攝影機
    if not cap.isOpened():
        print("無法開啟攝影機")
        return None

    # 設定影片格式與輸出檔案
    print("攝影機解析度:", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 視訊編碼格式
    # 檔名, 格式, FPS, 畫面尺寸
    output_path = os.path.join('results', f'{userid}.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    start_time = time.time()
    while time.time() - start_time < seconds:
        if stop_flags_dict.get(userid, False):
            print("偵測到停止指令")
            break

        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
            break

        out.write(frame)

        # 顯示畫面
        #cv2.imshow('Webcam', frame)

        # 按下 q 鍵離開
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("webcam完成錄影")
    return f".//{output_path}"
