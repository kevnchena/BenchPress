from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware #前端讀取
import uuid
import os
import threading
from pydantic import BaseModel

from webcam import webcam_on  # webcam錄影
from benchpress_analyzer import analyze_video # 原本的分析主程式
from yoloSeg import BenchPress_Seg

import mysql.connector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # CRA
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,        # 若有帶 cookie/credential 時需要
    allow_methods=["*"],           # 讓 OPTIONS/POST/GET… 都通過
    allow_headers=["*"],           # 讓 Content-Type/Authorization 等表頭通過
)

#檔案位置
UPLOAD_DIR = "temp_videos"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#錄影全域變數
recording_threads = {}
stop_flags = {}

#sql使用
class User(BaseModel):
    userid: str
    email: str


#主要分析
@app.post("/record")
def record_and_analyze(user:User,background_tasks: BackgroundTasks):
    user_id = user.userid
    stop_flags[user_id] = False #停止錄影flags
    print(user_id)

    video_path = os.path.join(UPLOAD_DIR, f"{user_id}.mp4")
    thread = threading.Thread(target=run_full_process, args=(user_id, video_path))
    thread.start()
    #background_tasks.add_task(run_full_process, user_id, video_path)
    return {"message": "錄影啟動", "user_id": user_id}

#停止錄影
@app.post("/stop/{userid}")
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
        json_path, analyzed_video_path = analyze_video(
            video_path, "L",
            f".//{OUTPUT_DIR}//{userid}_analyzed.mp4",
            f".//{OUTPUT_DIR}//{userid}.json")

    # Step 4: 可擴充回傳 or 存資料
        print(f"分析完成！json: {json_path}, Video: {analyzed_video_path}")
        return FileResponse(json_path, media_type="application/json")
    else:
        print(f"沒有錄影檔案，跳過分析 {userid}")
        return {"error": "<UNK>"}

#下載影片
@app.get("/results/video/{userid}")
def get_video(userid: str):
    path = os.path.join(OUTPUT_DIR, f"{userid}_analyzed.mp4")
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4", filename=f"{userid}_result.mp4")
    else:
        return {"error": f"找不到{path}檔案"}

#下載json
@app.get("/results/json/{userid}")
def get_csv(userid: str):
    path = os.path.join(OUTPUT_DIR, f"{userid}.json")
    print(path)
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json")
    else:
        return {"error": "找不到json檔案"}

#------SQL------

#插入使用者uid、email
@app.post("/sql")
def signup_to_sql(user: User):
    con = mysql.connector.connect(
        user="root",
        password="6256875",
        host="localhost",
        database="mydb"
    )
    cursor = con.cursor()
    cursor.execute("INSERT INTO users(uid, email) VALUES (%s, %s)",(user.userid, user.email))
    con.commit()
    con.close()
    print(f"user.id: {user.userid},user.email: {user.email}")
    return {"status": "ok", "uid": user.userid, "email": user.email}

#插入臥推資料
@app.post("/sql")
def signup_to_sql(user: User):
    con = mysql.connector.connect(
        user="root",
        password="6256875",
        host="localhost",
        database="mydb"
    )
    cursor = con.cursor()
    cursor.execute("INSERT INTO users(uid, email) VALUES (%s, %s)",(user.userid, user.email))
    con.commit()
    con.close()
    print(f"user.id: {user.userid},user.email: {user.email}")
    return {"status": "ok", "uid": user.userid, "email": user.email}


