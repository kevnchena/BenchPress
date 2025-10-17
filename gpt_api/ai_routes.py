from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

router = APIRouter()

class AnalyzeRequest(BaseModel):
    reps: list  # 前端送來的 rep JSON
    avg: dict   # 也可以送 avgPower, avgEccentric 等 summary

@router.post("/gpt/analyze")
def gpt_analyze(data: AnalyzeRequest):
    prompt = f"""
    這是一組臥推動作數據，請逐一下 rep 做簡短分析，
    並在最後提供總體評估與建議。請用簡潔專業的教練語氣輸出。

    數據 JSON:
    {data.dict()}
    """

    resp = client.responses.create(
        model="gpt-4o-mini",
        temperature=0.2,
        input=[
            {
                "role": "system",
                "content": "你是一位專業健身教練，會針對 bench press 數據做分析，主要給予力量和組間休息的建議。請把這組訓練的每一下做出分析並且給整組一個評語。push_dips表示該組的向心是否有下沉。"

            },


            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 把 GPT 輸出的文字回傳
    return {"summary": resp.output[0].content[0].text}