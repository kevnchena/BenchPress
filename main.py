from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import json

app = FastAPI()

# 靜態資料庫：可被 GPT 工具讀取的臥推分析資料
class BenchAnalysis(BaseModel):
    video_id: str
    user_id: str
    avg_arm_angle: float
    bounce: bool
    label: str
    gpt_feedback: str

# 假資料庫
STATIC_DATA = [
    BenchAnalysis(
        video_id="user01_rep1",
        user_id="user01",
        avg_arm_angle=57.8,
        bounce=True,
        label="手肘過度外展",
        gpt_feedback="肘部外展角度過大，建議控制在 60 度內，並避免彈槓技巧。"
    ),
    BenchAnalysis(
        video_id="user02_rep3",
        user_id="user02",
        avg_arm_angle=64.2,
        bounce=False,
        label="動作標準",
        gpt_feedback="手臂控制良好，動作穩定，建議持續維持。"
    )
]

@app.get("/get-analysis", response_model=BenchAnalysis)
def get_analysis(video_id: str):
    for item in STATIC_DATA:
        if item.video_id == video_id:
            return item
    return {"error": "查無資料"}

@app.get("/all-analyses", response_model=List[BenchAnalysis])
def all_analyses():
    return STATIC_DATA

@app.get("/")
def root():
    return {"message": "靜態臥推分析資料庫 API"}
