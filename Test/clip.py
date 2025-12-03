from moviepy import VideoFileClip
import json, os

uid = "wfqVHUZO7tSpON7TLv04egV1L1o2"
base_dir = "../output"
video_path = os.path.join(base_dir, f"{uid}_analyzed.mp4")
meta_path = os.path.join(base_dir, f"{uid}_meta.json")

# 讀取 meta.json
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

segments = meta.get("abnormal_segments", [])
if not segments:
    print("沒有異常片段可剪接")
    exit()

# 剪接全部異常區段
for seg in segments:
    start, end = seg["start_time"], seg["end_time"]
    rep, seg_type = seg["rep"], seg["type"]
    out_path = os.path.join(base_dir, f"clip_{uid}_rep{rep}_{seg_type}.mp4")

    print(f"🎞️ 剪接 Rep {rep} ({seg_type}) [{start}→{end}]")
    clip = VideoFileClip(video_path).subclipped(start, end)
    clip.write_videofile(out_path, codec="libx264", audio=False)

print("✅ 全部異常片段已輸出！")