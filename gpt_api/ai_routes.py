from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
import json

client = OpenAI()
router = APIRouter()


class AnalyzeRequest(BaseModel):
    reps: list  # 前端送來的 rep JSON
    avg: dict   # 可選的 summary 資料


@router.post("/gpt/analyze")
async def gpt_analyze(data: AnalyzeRequest):
    """
    使用 GPT-5-mini 分析 bench press：
    - 每個 rep 都要有分數與評論
    - 最後要有總結與建議
    - 輸出為乾淨 JSON 格式
    """

    prompt = f"""
You are a professional strength and conditioning coach.
Analyze the following bench press data (each rep includes timing, power, and stability metrics).

For each rep:
- Provide a short evaluation in Traditional Chinese.
- Give a numeric score from 1 to 10 (higher = better execution).

At the end, provide:
- An overall summary in Traditional Chinese (covering tempo, stability, fatigue).
- A training recommendation (e.g., rest duration or load adjustment).

Required JSON format (must be valid JSON, no explanation text):
{{
  "reps_analysis": [
    {{"rep": 1, "score": 8.5, "comment": "動作穩定且速度理想。"}},
    {{"rep": 2, "score": 7.0, "comment": "離心略快，需加強控制。"}}
  ],
  "overall_summary": "整體節奏良好，後段出現疲勞跡象。",
  "training_advice": "建議組間休息 3～4 分鐘。"
}}

Analyze this data:
{json.dumps(data.dict(), ensure_ascii=False)}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            #temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert fitness coach. "
                        "Always respond in valid JSON only, no other text, comments, or formatting."
                        "All comments must be written in Traditional Chinese."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw_output = resp.choices[0].message.content.strip()
        cleaned = raw_output.replace("\n", "").replace("\r", "").strip()

        # 嘗試解析 JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = {"error": "Invalid JSON output", "raw": cleaned}
        print(resp.choices[0].message.content.strip())
        return {"summary": parsed}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"GPT分析失敗: {str(e)}"}