from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
import uuid
import os
import threading

from webcam import webcam_on  # webcam錄影
from benchpress_analyzer import analyze_video # 原本的分析主程式
from yoloSeg import BenchPress_Seg

app = FastAPI()

#檔案位置
UPLOAD_DIR = "temp_videos"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#錄影全域變數
recording_threads = {}
stop_flags = {}

#主要分析
@app.post("/record")
def record_and_analyze(background_tasks: BackgroundTasks):
    user_id = str(uuid.uuid4())
    stop_flags[user_id] = False #停止錄影flags
    print(user_id)

    video_path = os.path.join(UPLOAD_DIR, f"{user_id}.mp4")
    thread = threading.Thread(target=run_full_process, args=(user_id, video_path))
    thread.start()
    #background_tasks.add_task(run_full_process, user_id, video_path)
    return {"message": "錄影與分析任務已啟動", "user_id": user_id}

#停止錄影
@app.post("/stop/userid")
def stop_recording(userid: str):
    if userid in stop_flags:
        stop_flags[userid] = True #停止錄影flags
        return {"message": f"已發出停止錄影指令：{userid}"}
    else:
        return {"error": "找不到這個使用者 ID 或錄影尚未開始"}

#全運行
def run_full_process(userid, video_path):
    # Step 1: 錄影
    video_path = webcam_on(userid, stop_flags)

    # Step 2: YOLO分割
    video_path = BenchPress_Seg(userid, video_path)
    print(video_path)

    # Step 3: 分析動作
    if os.path.exists(video_path):
        csv_path, analyzed_video_path = analyze_video(
            video_path, "L",
            f".//{OUTPUT_DIR}//{userid}_analyzed.mp4",
            f".//{OUTPUT_DIR}//{userid}.csv")

    # Step 4: 可擴充回傳 or 存資料
        print(f"分析完成！CSV: {csv_path}, Video: {analyzed_video_path}")
    else:
        print(f"沒有錄影檔案，跳過分析 {userid}")

#下載影片
@app.get("/results/video/{userid}")
def get_video(userid: str):
    path = os.path.join(OUTPUT_DIR, f"{userid}_analyzed.mp4")
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4", filename=f"{userid}_result.mp4")
    else:
        return {"error": f"找不到{path}檔案"}

#下載csv
@app.get("/results/csv/{userid}")
def get_csv(userid: str):
    path = os.path.join(OUTPUT_DIR, f"{userid}.csv")
    print(path)
    if os.path.exists(path):
        return FileResponse(path, media_type="text/csv")
    else:
        return {"error": "找不到csv檔案"}

