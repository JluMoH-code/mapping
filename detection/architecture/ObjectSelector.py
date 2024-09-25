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
        self.window = 'Frame'

    def select_object(self, detections: List[Detection], position) -> Optional[Detection]:
        self.selected_object = self.find_nearest_object(detections, position)
        if self.selected_object:
            return True
        try:
            self.selected_object = self.select_square_area(position)
            return True
        except:
            return False

    def find_nearest_object(self, detections, position) -> Optional[Detection]:
        min_distance = float('inf')
        nearest_detection = None
        
        for detection in detections:
            box = detection.box
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            distance = (center_x - position[0]) ** 2 + (center_y - position[1]) ** 2
            if distance < min_distance:
                min_distance = distance
                nearest_detection = detection
        
        return nearest_detection
    
    def select_square_area(self, position, min_area=(100, 100)):
        x1 = position[0] - min_area[0] // 2
        y1 = position[1] - min_area[0] // 2
        x2 = position[0] + min_area[0] // 2
        y2 = position[1] + min_area[1] // 2
        box = (x1, y1, x2, y2)

        return Detection(box=box)