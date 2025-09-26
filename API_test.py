import time

from fastapi import FastAPI, BackgroundTasks,Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware #前端讀取
from typing import Optional

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
done = threading.Event()

#sql使用
class Route(BaseModel):
    userid: str
    weight: Optional[float] = 0


#主要分析
@app.post("/record")
def record_and_analyze(user:Route,background_tasks: BackgroundTasks):
    userid = user.userid
    weight = float(user.weight or 0)
    stop_flags[userid] = False #停止錄影flags
    print(userid)
    print(user.weight)

    video_path = os.path.join(UPLOAD_DIR, f"{userid}.mp4")
    thread = threading.Thread(target=run_full_process, args=(userid, weight , video_path, done))
    thread.start()
    #background_tasks.add_task(run_full_process, user_id, video_path)
    return {"message": "錄影啟動", "user_id": userid}

#停止錄影
@app.post("/stop")
def stop_recording(user:Route):
    userid = user.userid
    weight = float(user.weight or 0)
    print(weight)

    json_path = os.path.join(OUTPUT_DIR, f"{userid}.json")

    if userid in stop_flags:
        stop_flags[userid] = True #停止錄影flags
        done.wait()
        print("分析完成，送出json")
        time.sleep(2)  #檔案建立煞車
        return FileResponse(json_path, media_type="application/json")
    else:
        done.wait()
        print("分析失敗")
        return {"error": "<UNK>"}

#全運行
def run_full_process(userid, weight, video_path, done_evt: threading.Event):
    # Step 1: 錄影
    video_path = webcam_on(userid, stop_flags)

    # Step 2: YOLO分割
    video_path = BenchPress_Seg(userid, video_path)

    # Step 3: 分析動作
    if os.path.exists(video_path):
        json_path, analyzed_video_path = analyze_video(
            video_path, "L",
            f".//{OUTPUT_DIR}//{userid}_analyzed.mp4",
            f".//{OUTPUT_DIR}//{userid}.json",
            weight=weight
        )

    # Step 4: 可擴充回傳 or 存資料
        done_evt.set()
        print(f"分析完成！json: {json_path}, Video: {analyzed_video_path}")
    else:
        done_evt.set()
        print(f"沒有錄影檔案，跳過分析 {userid}")

#下載影片
@app.get("/results/video")
def get_video(uid: str = Query(..., description="Firebase UID")):
    path = os.path.join(OUTPUT_DIR, f"{uid}_analyzed.mp4")
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4", filename=f"{uid}_result.mp4")
    else:
        return {"error": f"找不到{path}檔案"}

#下載json
@app.get("/results/json")
def get_csv(uid: str = Query(..., description="Firebase UID")):
    path = os.path.join(OUTPUT_DIR, f"{uid}.json")
    print(path)
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json")
    else:
        return {"error": "找不到json檔案"}

#------SQL------

#sql使用
class User(BaseModel):
    userid: str
    email: str


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


