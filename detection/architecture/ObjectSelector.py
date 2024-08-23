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
        self.update_flag = False

    def select_object(self, detections: List[Detection]) -> Optional[Detection]:
        self.detections = detections
        cv2.setMouseCallback(self.window, self.on_click)

        if self.update_flag:
            self.update_flag = False
            return self.selected_object

        return None

    def on_click(self, event, x, y, flags, param):
        nearest_object = None
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_position = (x, y)
            nearest_object = self.find_nearest_object()

        if nearest_object:
            self.selected_object = nearest_object
            self.update_flag = True

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
