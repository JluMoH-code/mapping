from ultralytics import YOLO 
import os
  
weights = r"F:\Всё\Linkos\navigation\detection\18k aero (batch 32)\weights\best.pt"
model = YOLO(weights)

basePath = r"F:\Всё\Linkos\navigation\detection\test_data"
imagePaths = [basePath + "\\" + imagePath for imagePath in os.listdir(basePath)]
model.predict(imagePaths, save=True)