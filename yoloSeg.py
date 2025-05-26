import cv2
from ultralytics import YOLO
import numpy as np
import torch
import os


# 載入 YOLOv11 模型
model = YOLO('..//best_seg.pt')

print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    model.to("cuda")
    print("使用中的 GPU 裝置名稱:", torch.cuda.get_device_name(0))
    print("目前裝置:", torch.cuda.current_device())
else:
    print("⚠️ 當前為 CPU-only 環境，無法使用 GPU！")

# 載入 YOLOv11 模型

def BenchPress_Seg(userid,video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        print(f"得到使用者路徑{video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path = f".//results//{userid}_result.mp4"

        # 建立影片輸出物件
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 偵測
            results = model(frame)[0]

            # 找出信心度最高的 bbox
            max_conf = 0.9
            best_bbox = None

            for box in results.boxes:
                conf = box.conf[0].item()
                if conf > max_conf:
                    max_conf = conf
                    best_bbox = box.xyxy[0].tolist()

            # 如果有找到目標
            if best_bbox:
                x1, y1, x2, y2 = map(int, best_bbox)
                # 建立黑白遮罩
                mask = cv2.rectangle(
                    np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8),
                    (x1, y1), (x2, y2), 255, -1
                )

                # 背景模糊
                blurred = cv2.GaussianBlur(frame, (55, 55), 0)

                # 將目標區域保持清晰
                foreground = cv2.bitwise_and(frame, frame, mask=mask)
                background = cv2.bitwise_and(blurred, blurred, mask=cv2.bitwise_not(mask))
                combined = cv2.add(foreground, background)

                # 顯示目標框
                cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(combined, f"Conf: {max_conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


            else:
                combined = frame

            out.write(combined)
            cv2.imshow("YOLOv11 Background Blur", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        os.remove(video_path)
        print(f"完成分割{output_path}")
        return output_path

    except:
        print(Exception)
        return None
