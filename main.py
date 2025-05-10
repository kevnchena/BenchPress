from fastapi import FastAPI, File, UploadFile
import shutil, os
from benchpress_analyzer import analyze_video  # 你原本的處理函式

app = FastAPI()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_path = f"temp_uploads/{file.filename}"

    # 儲存上傳影片
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 呼叫分析邏輯
    result_csv_path, result_video_path = analyze_video(temp_path)

    return {
        "message": "分析完成！",
        "csv_path": result_csv_path,
        "video_path": result_video_path
    }