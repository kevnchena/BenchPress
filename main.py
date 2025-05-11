from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid, os
from benchpress_analyzer import analyze_video  # 你原本的分析主程式，需輸出 CSV 路徑與影片路徑
from firebase_admin import credentials, initialize_app, firestore
import shutil

# ==== 初始化 Firebase Firestore ====
cred = credentials.Certificate("firebase_key.json")  # 放你的 Firebase 私鑰路徑
initialize_app(cred)
db = firestore.client()

# ==== FastAPI app ====
app = FastAPI()
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ==== 背景分析任務 ====
def background_analyze(video_path: str, user_id: str, session_id: str):
    csv_path, vis_path = analyze_video(video_path, session_id)

    # 將結果存進 Firebase（可擴充）
    doc_ref = db.collection("benchpress_sessions").document(session_id)
    doc_ref.set({
        "user_id": user_id,
        "session_id": session_id,
        "video_path": video_path,
        "csv_path": csv_path,
        "vis_path": vis_path,
    })


# ==== 分析 API（接收影片、排程分析） ====
@app.post("/analyze")
async def upload_video(user_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 用 UUID 命名避免重複
    session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}.mp4")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 加入背景任務
    background_tasks.add_task(background_analyze, save_path, user_id, session_id)

    return JSONResponse({
        "message": "已收到影片並排入分析任務",
        "session_id": session_id
    })


# ==== 查詢分析結果 ====
@app.get("/result/{session_id}")
def get_result(session_id: str):
    doc = db.collection("benchpress_sessions").document(session_id).get()
    if doc.exists:
        return doc.to_dict()
    return {"error": "Session not found"}


# ==== 健康檢查 ====
@app.get("/")
def root():
    return {"message": "BenchPress Analyzer API is running!"}
