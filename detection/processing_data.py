from typing import List
import ijson
import matplotlib.pyplot as plt
import numpy as np

# Класс для хранения координат ограничивающего прямоугольника
class BoundingBox:
    def __init__(self, bbox: List[float], center_bbox: List[float]):
        self.bbox = bbox
        self.center_bbox = center_bbox

# Класс для хранения данных об обнаруженных объектах
class DetectedObject:
    def __init__(self, class_id: int, bbox: BoundingBox, confidence: float):
        self.class_id = class_id
        self.bbox = bbox
        self.confidence = confidence

# Класс для хранения данных о кадре, включая отслеживаемый объект и обнаруженные объекты
class TrackedObject:
    def __init__(self, frame_number: int, is_tracking: bool, tracked_bbox: BoundingBox, detected_bboxes: List[DetectedObject]):
        self.frame_number = frame_number
        self.is_tracking = is_tracking
        self.tracked_bbox = tracked_bbox
        self.detected_bboxes = detected_bboxes
        
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        
# Функция для парсинга JSON в список объектов
def parse_json_object(frame_data: dict) -> List[TrackedObject]:
    tracked_objects = []
    
    frame_number = frame_data['frame_number']
    is_tracking = frame_data['is_tracking']
    
    # Создаем объект BoundingBox для отслеживаемого объекта
    tracked_bbox = BoundingBox(
        bbox=frame_data['tracked_bbox'],
        center_bbox=frame_data['center_tracked_bbox']
    )
    
    # Создаем список объектов DetectedObject
    detected_bboxes = []
    for detected in frame_data['detected_bboxes']:
        detected_bbox = BoundingBox(
            bbox=detected['bbox'],
            center_bbox=detected['center_bbox']
        )
        detected_obj = DetectedObject(
            class_id=detected['class_id'],
            bbox=detected_bbox,
            confidence=detected['confidence']
        )
        detected_bboxes.append(detected_obj)
    
    # Создаем объект TrackedObject и добавляем его в список
    tracked_object = TrackedObject(
        frame_number=frame_number,
        is_tracking=is_tracking,
        tracked_bbox=tracked_bbox,
        detected_bboxes=detected_bboxes
    )
    
    return tracked_object

def read_json_stream(file_path: str) -> List[TrackedObject]:
    tracked_objects = []
    
    with open(file_path, 'r') as file:
        objects = ijson.items(file, 'item')
        for obj in objects:
            tracked_object = parse_json_object(obj)
            tracked_objects.append(tracked_object)
    
    return tracked_objects

def visualize_center_bboxes(tracked_objects):
    x_coords = []
    y_coords = []

    for obj in tracked_objects:
        x, y = obj.tracked_bbox.center_bbox
        if (x == 0 and y == 0): continue  # Отсекаем нулевые координаты
        x_coords.append(x)
        y_coords.append(y)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b', label='Center BBox Path')
    
    # Оформление графика
    plt.title('Tracking Center BBox Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def match_tracked_to_detected(tracked_objects: List[TrackedObject], distance_threshold: float = 5.0):
    matches = []
    count = 0

    for obj in tracked_objects:
        tracked_center = obj.tracked_bbox.center_bbox
        matched_detected = None
        min_distance = float('inf')
        
        for detected in obj.detected_bboxes:
            detected_center = detected.bbox.center_bbox
            distance = euclidean_distance(tracked_center, detected_center)
            
            if distance < min_distance and distance <= distance_threshold:
                min_distance = distance
                matched_detected = detected
        
        if matched_detected:
            matches.append((obj.frame_number, tracked_center, matched_detected.bbox.center_bbox, min_distance))
            count += 1
        else:
            matches.append((obj.frame_number, tracked_center, None, None))
            
    print(count)
    return matches

file_path = "results.json"
tracked_objects = read_json_stream(file_path)
matches = match_tracked_to_detected(tracked_objects, distance_threshold=25.0)
# visualize_center_bboxes(tracked_objects)