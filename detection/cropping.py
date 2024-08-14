import cv2
import os
import re

save_frames = True

def save_frame(video_path, save_path, start_frame, save_interval=1):
    global save_frames
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл.")
        return
    
    frame_count = 0
    saved_frame_count = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        frame_count += 1
        
        frame_show = cv2.resize(frame, (1024, 768))
        cv2.imshow('Video', frame_show)
        
        key = cv2.waitKey(1)
        if key == ord(' '):
            save_frames = not save_frames
        
        if save_frames:
            if frame_count % save_interval == 0:
                saved_frame_count += 1
                filename = f'frame_{saved_frame_count}.jpg'
                cv2.imwrite(save_path + filename, frame)

   
save_path = "train_data\\cropped_images\\"
video_path = "train_data\\videos\\crop_2023_03_19_22_03_24.mp4"
start_frame = 0

images = os.listdir(save_path)         
if images:
    for image in images:
        number = re.search("[0-9]+", image)
        if int(number.group()) > start_frame:
            start_frame = int(number.group())

save_frame(video_path, save_path, start_frame, 30)               