from ultralytics import YOLO 
import os
  
weights = r"trained_models\18k aero (batch 32)\weights\best.pt"
model = YOLO(weights)

basePath = r"test_data\\"
imagePaths = [basePath + imagePath for imagePath in os.listdir(basePath)]
model.predict(
    imagePaths, 
    save=True,
    iou=0.4,
    show_labels=True,
    show_conf=False,
    )