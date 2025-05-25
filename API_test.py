from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
import uuid
import os

from webcam import webcam_on  # ✅ 你自己的錄影 function
from benchpress_analyzer import analyze_video      # ✅ 你原本的分析主程式

app = FastAPI()

UPLOAD_DIR = "temp_videos"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/record")
def record_and_analyze(background_tasks: BackgroundTasks):
    user_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{user_id}.mp4")
    background_tasks.add_task(run_full_process, user_id, video_path)
    return {"message": "錄影與分析任務已啟動", "user_id": user_id}


def run_full_process(userid, video_path):
    # ✅ Step 1: 錄影
    video_path = webcam_on(userid)

    # ✅ Step 2: 分析
    csv_path, analyzed_video_path = analyze_video(
        video_path, "L", f".//{OUTPUT_DIR}//{userid}_analyzed.mp4",
                                    f".//{OUTPUT_DIR}//{userid}.csv")

    # ✅ Step 3: 可擴充回傳 or 存資料
    print(f"分析完成！CSV: {csv_path}, Video: {analyzed_video_path}")


@app.get("/results/video/{userid}")
def get_video(userid: str):
    path = os.path.join(OUTPUT_DIR, f"{userid}_analyzed.mp4")
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4", filename=f"{userid}_result.mp4")
    else:
        return {"error": f"找不到{path}檔案"}


@app.get("/results/csv/{userid}")
def get_csv(userid: str):
    path = os.path.join(OUTPUT_DIR, f"{userid}.csv")
    print(path)
    if os.path.exists(path):
        return FileResponse(path, media_type="text/csv")
    else:
        return {"error": "找不到csv檔案"}

