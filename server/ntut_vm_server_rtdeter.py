####################################################################################
#############################################################
# -*- coding=utf-8 -*-
__author__="ITRI-EOSL-R300 A30335-Rachel"
# NTUT_2025 CV
# env: teacher_cv
# ps -ef | grep nsr
##############################################################
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import sys
import glob
from typing import List
import uvicorn
from datetime import datetime
from ntut_rtdeter_inf import DETR_inf

'''
source deactivate
pip install fastapi uvicorn
pip install python-multipart
'''

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
print(root_dir)
UPLOAD_DIR = os.path.join(root_dir, "API/RTDETER_uploads")
OUTPUT_DIR = os.path.join(root_dir, "API/RTDETER_outputs")


# 確保必要的目錄存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR2=OUTPUT_DIR
MODEL='NTUT_RTDETER.pt'
MODEL_P = os.path.join(UPLOAD_DIR, MODEL)
@app.post("/upload-model/")
async def upload_model(file: UploadFile = File(...)):
    #timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_file_path = os.path.join(UPLOAD_DIR, MODEL)
    if(os.path.exists(model_file_path)):
        os.remove(model_file_path)
    with open(model_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(model_file_path) 
    return os.path.basename(model_file_path)

@app.post("/upload-img/{modelp}")
async def upload_img(file: UploadFile = File(...)):
    try:
        # 建立唯一的輸出目錄名稱（使用檔案名稱去除副檔名）
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename_without_ext = os.path.splitext(file.filename)[0]
        baseF = f'{timestamp}_{filename_without_ext}'
        dst_folder_new = os.path.join(OUTPUT_DIR,baseF )
        OUTPUT_DIR2 = dst_folder_new
        os.makedirs(dst_folder_new, exist_ok=True)

        # 儲存上傳的 image 檔案
        img_file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(img_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 呼叫 PDF 切割函式
        #image_paths = pdf_to_images(pdf_path=pdf_file_path, output_folder=dst_folder_new)
            
        # 對每個圖片執行 YOLOv7
        #for image_path in image_paths:
        summary_file_path, time_path = DETR_inf(model_path=MODEL_P,img_path=img_file_path, savedir=dst_folder_new)
        
        # 取得處理後的圖片列表
        processed_images = [i for i in glob.glob(os.path.join(dst_folder_new, '*.jpg'))]
        
        # 回傳處理結果
        return {
            "message": "IMAGE 處理成功",
            "filename": file.filename,
            "processed_images": processed_images,
            "summary_file_path": summary_file_path,
            "time_path": time_path,
            "total_images": len(processed_images),
            "OUTPUT_DIR2": OUTPUT_DIR2,
            "baseF":baseF
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/get-images/{img_name}")
async def get_images(img_name: str):
    try:
        # 使用 glob 取得所有 PNG 圖片
        #image_dir = os.path.join(OUTPUT_DIR, img_name.replace(".png", "").replace(".jpg", ""))
        #image_list = [os.path.basename(i) for i in glob.glob(os.path.join(OUTPUT_DIR2, '*.jpg'))]
        
        # 回傳圖片檔案的檔名清單
        return {"images": img_name, "total": len(list(img_name))}
    except Exception as e:
        return {"error": str(e)}

@app.get("/download-image/{file_name}/{dir}")
async def download_image(file_name: str, dir: str):
    try:
        if(dir):
            image_path = os.path.join(os.path.join(OUTPUT_DIR, dir),file_name)
        else:
            image_path = os.path.join(OUTPUT_DIR,file_name)
        print('-'*20,image_path )
        return FileResponse(image_path)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)