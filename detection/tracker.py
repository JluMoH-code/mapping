from ultralytics import YOLO 
import cv2
import os
from time import time
import numpy as np
from dotenv import load_dotenv
load_dotenv() 

weights = os.getenv("WEIGHTS_PATH")
model = YOLO(weights)

videoPath = os.getenv("VIDEO_PATH")
videoCap = cv2.VideoCapture(videoPath)
ok, frame = videoCap.read()
model(frame)

def dist_to_xy(center_xy, x2, y2):
    x1 = center_xy[0]
    y1 = center_xy[1]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def resize_image(frame):
    height, width = frame.shape[:2]  # Получаем высоту и ширину изображения

    if height <= 720 and width <= 720:
        # Если размеры меньше или равны 720, увеличиваем в 10 раз
        new_width = width * 10
        new_height = height * 10
    else:
        # Иначе, уменьшаем изображение
        scaling_factor = 720 / max(height, width)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
    
    resized_frame = cv2.resize(frame, (new_width, new_height))  # Изменяем размер изображения
    return resized_frame

def detect_objects(frame):
    results = model(frame)
    return results[0].boxes.cpu().numpy()

def find_closest_bbox(boxes, target_point):
    min_dist = float('inf')
    best_bb = None

    for box in boxes:
        r_xyxy = box.xyxy[0].astype(int)
        x1, y1, x2, y2 = r_xyxy
        current_center = ((x2 + x1) // 2, (y2 + y1) // 2)
        
        distance = dist_to_xy(target_point, current_center[0], current_center[1])
        if distance < min_dist:
            min_dist = distance
            best_bb = [x1, y1, x2 - x1, y2 - y1]  # Формат (x, y, w, h)

    return best_bb

def get_bbox(frame, click_x, click_y):
    start_time = time()
    boxes = detect_objects(frame)

    closest_bbox = find_closest_bbox(boxes, (click_x, click_y))

    print("Time to find car:", time() - start_time)
    print("Best bbox:", closest_bbox)

    if closest_bbox:
        x, y, w, h = closest_bbox
        frame_cropped = frame[y:y + h, x:x + w]
        resized_frame = resize_image(frame_cropped)
        cv2.imshow("Detected Object", resized_frame)
        return closest_bbox
    return None

def init_tracker(frame, bbox):
    # Инициализируем трекер с полученным bbox
    global tracker, tracker_init, tracker_type
    
    tracker_init = True
    
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()                   # хорошая точность и неплохая скорость работы
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()     # великолепная скорость работы (~60 fps)
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()  
    elif tracker_type == 'MOSSE':                           # Minimum Output Sum of Squared Error
        tracker = cv2.legacy.TrackerMOSSE_create()          # невероятная скорость работы и неплохая точность (~60fps)
    
    return tracker.init(frame, bbox)

def mouseClick(event, x, y, flags, param):
    global bbox, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = get_bbox(frame, x, y)
        if bbox is not None:
            if not init_tracker(frame, bbox): 
                print("Error initialize tracker!")
        else:
            print("BBox not found")

def yolo_autodetect(frame, last_known_bbox=None):
    global tracker
    boxes = detect_objects(frame)

    if last_known_bbox:
        x, y, w, h = last_known_bbox
        last_center = (x + w // 2, y + h // 2)
        closest_bbox = find_closest_bbox(boxes, last_center)

        if closest_bbox:
            init_tracker(frame, closest_bbox)
            return closest_bbox
    
    return None
    
def draw_bbox(frame, bbox, color=(255, 0, 0), thickness=2):
    """Рисует bounding box на кадре."""
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

def show_tracked_object(frame, bbox, window_name="Tracked Object"):
    """Показывает отслеживаемый объект в отдельном окне."""
    x, y, w, h = map(int, bbox)
    tracked_frame = frame[y:y + h, x:x + w]
    cv2.imshow(window_name, tracked_frame)

def process_tracking(frame, bbox):
    """Обрабатывает трекинг и корректирует его при необходимости."""
    global tracker_init, last_frame_bbox, tracker

    if tracker_init:
        ok, bbox = tracker.update(frame)
        if ok:
            draw_bbox(frame, bbox)
            show_tracked_object(frame, bbox)
            last_frame_bbox = bbox
        else:
            tracker_init = False
            last_frame_bbox = yolo_autodetect(frame, last_frame_bbox)

def should_update_tracker(last_detection_time, interval=1.0):
    """Определяет, нужно ли обновить трекер на основе времени."""
    return time() - last_detection_time >= interval
    
cv2.namedWindow("Main")
cv2.setMouseCallback("Main", mouseClick)

bbox = (0, 0, 0, 0)
tracker_init = False
    
# Задаём тип трекера
tracker_type = 'MOSSE'
tracker = None

last_detection_time = time()
last_frame_bbox = None

while True:
    ret, frame = videoCap.read()
    if not ret:
        break

    process_tracking(frame, last_frame_bbox)

    if should_update_tracker(last_detection_time):
        last_frame_bbox = yolo_autodetect(frame, last_frame_bbox)
        last_detection_time = time()

    cv2.imshow('Main', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()