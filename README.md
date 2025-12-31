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
main.py為FastAPI主程式
所有使用功能有分類在不同資料夾中
fullprocess為主分析pipline
