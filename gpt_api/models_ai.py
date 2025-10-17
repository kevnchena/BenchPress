from pydantic import BaseModel
from typing import List, Optional

class CoachReq(BaseModel):
    uid: str
    notes: Optional[str] = None  # 前端想補充的備註（可選）

class RepAdvice(BaseModel):
    rep: int
    flags: List[str]              # e.g. ["unstable_velocity","long_bottom_pause"]
    cues: List[str]               # e.g. ["brace core","control descent"]

class CoachResp(BaseModel):
    session_summary: str          # 組總評（1~3 段）
    rest_seconds: int             # 建議休息秒數（例如 180~300）
    key_focus: List[str]          # 此組下一次的 3 個重點
    per_rep: List[RepAdvice]      # 逐一下建議（可只給重要的 rep）