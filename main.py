import time
from moviepy import VideoFileClip
import hashlib

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware #前端讀取
from gpt_api.ai_routes import router as ai_router

import os
from datetime import datetime
import json
import uuid
import threading
from pydantic import BaseModel
from typing import Optional, List, Dict

from fullprocess.webcam import webcam_on  # webcam錄影
from fullprocess.benchpress_analyzer import analyze_video # 原本的分析主程式
from fullprocess.yoloSeg import BenchPress_Seg

import mysql.connector

app = FastAPI()
app.include_router(ai_router)

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
META_DIR = "./output/meta"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


#錄影全域變數
#recording_threads = {}
stop_flags = {}
done = threading.Event()

# 主頁面使用
class Route(BaseModel):
    userid: str
    weight: Optional[float] = 0


# 主要分析
@app.post("/record")
def record_and_analyze(user:Route):
    userid = user.userid
    weight = float(user.weight or 0)
    stop_flags[userid] = False #停止錄影flags
    print(userid)
    print(user.weight)

    video_path = os.path.join(UPLOAD_DIR, f"{userid}.mp4")
    thread = threading.Thread(target=run_full_process, args=(userid, weight , video_path, done))
    thread.start()
    return {"message": "錄影啟動", "user_id": userid}

# 停止錄影
@app.post("/stop")
def stop_recording(user:Route):
    userid = user.userid
    weight = float(user.weight or 0)
    print(weight)

    json_path = os.path.join(OUTPUT_DIR, f"{userid}.json")
    #return FileResponse(json_path, media_type="application/json")
    if userid in stop_flags:
        stop_flags[userid] = True #停止錄影flags
        done.wait()
        print("分析完成，送出json")
        time.sleep(2)  #等待檔案建立安全時間
        return FileResponse(json_path, media_type="application/json")
    else:
        done.wait()
        print("分析失敗")
        return {"error": "<UNK>"}

# 全運行
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

# 下載影片
@app.get("/results/video")
def get_video(uid: str = Query(..., description="Firebase UID")):
    path = os.path.join(OUTPUT_DIR, f"{uid}_analyzed.mp4")
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4", filename=f"{uid}_result.mp4")
    else:
        return {"error": f"找不到{path}檔案"}

# 下載json
@app.get("/results/json")
def get_json(uid: str = Query(..., description="Firebase UID")):
    path = os.path.join(OUTPUT_DIR, f"{uid}.json")
    print(path)
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json")
    else:
        return {"error": "找不到json檔案"}

# 獲得異常.json
@app.get("/results/meta")
def get_meta(uid: str = Query(..., description="Firebase UID")):
    meta_path = os.path.join(OUTPUT_DIR, f"{uid}_meta.json")
    if os.path.exists(meta_path):
        return FileResponse(meta_path, media_type="application/json")
    else:
        return {"abnormal_segments": []}

# 異常影片片段
@app.get("/video/clip")
def get_video_clip(uid: str, start: float, end: float):
    """
    回傳影片指定時間區段 (mp4)
    自動快取，不會重複切片
    """
    try:
        video_path = os.path.join(OUTPUT_DIR, f"{uid}_analyzed.mp4")
        if not os.path.exists(video_path):
            return {"error": f"Video for {uid} not found"}

        # 進行 padding 避免切得太短
        pad_start = max(0, start - 0.3)
        pad_end = end + 0.3

        # 用 uid/start/end 做 hash
        key_raw = f"{uid}_{pad_start:.2f}_{pad_end:.2f}"
        key = hashlib.md5(key_raw.encode()).hexdigest()[:12]

        out_path = os.path.join(META_DIR, f"clip_{key}.mp4")

        # clip 已存在 → 不重切
        if os.path.exists(out_path):
            print("⚡ 使用快取 clip:", out_path)
            return FileResponse(out_path, media_type="video/mp4")

        # clip 不存在 → 切一次
        print("✂ 正在切 clip:", pad_start, pad_end)
        clip = VideoFileClip(video_path).subclipped(pad_start, pad_end)
        clip.write_videofile(out_path, codec="libx264", audio=False)
        clip.close()

        print("✅ 新增 clip:", out_path)
        return FileResponse(out_path, media_type="video/mp4")

    except Exception as e:
        return {"error": str(e)}

# history頁面session載入
@app.get("/sql/sessions")
def get_sessions(uid: str):
    con = mysql.connector.connect(
        user="root",
        password="6256875",
        host="localhost",
        database="mydb"
    )
    cur = con.cursor(dictionary=True)

    cur.execute("""
        SELECT id, uid, started_at, notes
        FROM sessions
        WHERE uid = %s
        ORDER BY started_at DESC
    """, (uid,))
    sessions = cur.fetchall()
    con.close()
    return {"sessions": sessions}

# 查詢rep
@app.get("/sql/session/{session_id}/reps")
def get_reps(session_id: int):
    con = mysql.connector.connect(
        user="root",
        password="6256875",
        host="localhost",
        database="mydb"
    )
    cur = con.cursor(dictionary=True)

    cur.execute("""
        SELECT *
        FROM benchpress_reps
        WHERE session_id = %s
        ORDER BY rep_no ASC
    """, (session_id,))
    reps = cur.fetchall()
    con.close()
    return {"reps": reps}

#=================SQL====================

# ------ sign_up ------
class User(BaseModel): #users database
    userid: str
    email: str

#插入users table
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

# ----- session table -----

class SessionCreate(BaseModel):
    uid: str
    notes: str | None = None
    avg_power: float | None = None
    avg_score: float | None = None
    total_reps: int | None = None

@app.post("/sql/session/create")
def create_session(data: SessionCreate):
    con = mysql.connector.connect(
        user="root",
        password="6256875",
        host="localhost",
        database="mydb"
    )
    cur = con.cursor()

    cur.execute("""
        INSERT INTO sessions (uid, started_at, notes)
        VALUES (%s, %s, %s)
    """, (data.uid, datetime.now(), data.notes or ""))
    con.commit()

    session_id = cur.lastrowid
    con.close()

    return {"status": "ok", "session_id": session_id}

# ----- bench_press table -----

class RepInsert(BaseModel):
    uid: str
    session_id: int
    reps: list

@app.post("/sql/reps/add")
def insert_reps(data: RepInsert):
    con = mysql.connector.connect(
        user="root",
        password="6256875",
        host="localhost",
        database="mydb"
    )
    cur = con.cursor()

    for r in data.reps:
        cur.execute("""
            INSERT INTO benchpress_reps (
              uid, session_id, rep_no, weight,
              y_high, y_low, depth_high, depth_low,
              eccentric_time, concentric_time, bottom_pause_time,
              speed_unstable, push_dips, power, score
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.uid,
            data.session_id,
            r.get("rep"),
            r.get("weight"),
            r.get("y_high"),
            r.get("y_low"),
            r.get("depth_high"),
            r.get("depth_low"),
            r.get("eccentric_time"),
            r.get("concentric_time"),
            r.get("bottom_pause_time"),
            int(r.get("speed_unstable", False)),
            int(r.get("push_dips", False)),
            r.get("power"),
            r.get("score")
        ))

    con.commit()
    con.close()

    return {"status": "ok", "inserted": len(data.reps)}

