import os
from ultralytics import YOLO
import torch
from dotenv import load_dotenv
load_dotenv()

devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(devices)
device = 0 if torch.cuda.is_available() else None
print(f"Using device: {device}")

path_data_yaml = os.getenv("BASE_PATH_TRAIN_DATA") + "/data.yaml"

epochs = os.getenv("TRAINING_EPOCHS")
batch = os.getenv("TRAINING_BATCH")

weights = os.getenv("WEIGHTS_PATH")
model = YOLO(weights)

results = model.train(
   data=path_data_yaml,
   epochs=epochs,
   batch=batch,
   device=device
   )