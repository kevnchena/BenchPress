import cv2
import os
import time


def webcam_on(userid, stop_flags_dict, seconds=60):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 強制 DirectShow
    print("Camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return None

    print("攝影機解析度:", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 編碼器：mp4v（要注意不同電腦支援度不同）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 輸出路徑
    os.makedirs('../results', exist_ok=True)
    output_path = os.path.join('../results', f'{userid}.mp4')

    print("儲存影片路徑:", output_path)

    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    print("VideoWriter opened:", out.isOpened())

    start_time = time.time()
    while time.time() - start_time < seconds:

        if stop_flags_dict.get(userid, False):
            print("🛑 偵測到停止指令")
            break

        ret, frame = cap.read()
        print("Frame read:", ret)

        if not ret:
            print("⚠️ 無法讀取畫面")
            break

        out.write(frame)  # 寫入影片

        cv2.imshow("Webcam Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 使用者手動退出")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("🎬 webcam完成錄影")

    return output_path


# ----------------------------------------
# ⭐ 測試程式入口
# ----------------------------------------
if __name__ == "__main__":
    print("=== Webcam-on 測試模式 ===")

    userid = "test_user"
    stop_flags_dict = {userid: False}

    print("開始錄影 10 秒...")
    video_path = webcam_on(userid, stop_flags_dict, seconds=10)

    print("\n錄影輸出:", video_path)

    if video_path and os.path.exists(video_path):
        size = os.path.getsize(video_path)
        print(f"影片大小: {size / 1024:.2f} KB")

        if size > 1000:
            print("✅ 影片看起來是正常的（大於 1MB）")
        else:
            print("❌ 影片太小，可能是空白的（失敗）")