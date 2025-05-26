使用前需先cmd
``pip install -r ./requirements.txt``
---
建立需要資料夾
``mkdir output,results,temp_videos``
---
Activate FastAPI
``uvicorn API_test:app --reload``
---
FastAPI test UI
http://localhost:8000/docs
---
userid 隨機產生，停止錄影、下載影片、下載 csv 檔案都需要 userid
