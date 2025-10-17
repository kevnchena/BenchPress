import json, os
from typing import List, Dict

OUTPUT_DIR = "../output"

def load_reps(uid: str) -> List[Dict]:
    path = os.path.join(OUTPUT_DIR, f"{uid}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []