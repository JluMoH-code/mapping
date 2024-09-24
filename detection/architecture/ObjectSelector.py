from typing import List, Optional, Any
from Detection import Detection
import cv2

class ObjectSelector:
    def select_object(self, detections: List[Detection], image: Any = None) -> Detection:
        raise NotImplementedError

class ClickObjectSelector(ObjectSelector):
    def __init__(self):
        self.selected_object = None
        self.click_position = None
        self.detections = []
        self.window = 'Frame'

    def select_object(self, detections: List[Detection], position) -> Optional[Detection]:
        self.detections = detections
        self.selected_object = self.find_nearest_object(position)
        if self.selected_object:
            return True
        return False

    def find_nearest_object(self, position) -> Optional[Detection]:
        min_distance = float('inf')
        nearest_detection = None
        
        for detection in self.detections:
            box = detection.box
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            distance = (center_x - position[0]) ** 2 + (center_y - position[1]) ** 2
            if distance < min_distance:
                min_distance = distance
                nearest_detection = detection
        
        return nearest_detection
