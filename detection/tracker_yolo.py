from ultralytics import YOLO 
import cv2
import os
from dotenv import load_dotenv
load_dotenv() 

weights = os.getenv("WEIGHTS_PATH")
model = YOLO(weights)

videoPath = os.getenv("VIDEO_PATH")
videoCap = cv2.VideoCapture(videoPath)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    results = model.track(frame, stream=True, persist=True)

    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                class_name = classes_names[cls]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()