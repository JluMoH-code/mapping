from ultralytics import YOLO 
import os
from dotenv import load_dotenv 
load_dotenv() 
  
weights = os.getenv("WEIGHTS_PATH")
model = YOLO(weights)

basePath = os.getenv("BASE_PATH_PHOTO")
imagePaths = [basePath + imagePath for imagePath in os.listdir(basePath)]
videoPath = os.getenv("VIDEO_PATH")

model.predict(
    videoPath, 
    save=True,
    iou=0.4,
    show_labels=True,
    show_conf=False,
    )