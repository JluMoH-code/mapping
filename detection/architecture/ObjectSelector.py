from typing import List, Optional, Any
from ObjectDetector import Detection
import cv2

class ObjectSelector:
    def select_object(self, detections: List[Detection], image: Any = None) -> Detection:
        raise NotImplementedError

class ClickObjectSelector(ObjectSelector):
    def __init__(self):
        self.selected_object = None
        self.click_position = None
        self.detections = None
        self.window = 'Frame'

    def select_object(self, detections: List[Detection], image: Any) -> Optional[Detection]:
        self.detections = detections
        cv2.setMouseCallback(self.window, self.on_click)
        return self.selected_object

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_position = (x, y)
            self.selected_object = self.find_nearest_object()

    def find_nearest_object(self) -> Optional[Detection]:
        min_distance = float('inf')
        nearest_detection = None
        
        for detection in self.detections:
            box = detection.box
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            distance = (center_x - self.click_position[0]) ** 2 + (center_y - self.click_position[1]) ** 2
            if distance < min_distance:
                min_distance = distance
                nearest_detection = detection
                
        return nearest_detection
