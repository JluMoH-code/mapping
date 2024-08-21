import cv2
from typing import Optional, Tuple, Any
from ObjectDetector import ObjectDetector, Detection

class ObjectTracker:
    def __init__(self, tracker_type: str = 'CSRT'):
        self.tracker = self.create_tracker(tracker_type)
        self.is_tracking = False
        self.bbox = None

    def create_tracker(self, tracker_type: str):
        if tracker_type == 'KLT':
            return cv2.TrackerKLT_create()
        elif tracker_type == 'MedianFlow':
            return cv2.TrackerMedianFlow_create()
        elif tracker_type == 'TLD':
            return cv2.TrackerTLD_create()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            return cv2.legacy.TrackerKCF_create()
        else:
            raise ValueError(f"Неизвестный тип трекера: {tracker_type}")

    def start_tracking(self, frame: Any, bbox: Tuple[int, int, int, int]) -> None:
        self.bbox = bbox
        self.is_tracking = self.tracker.init(frame, bbox)

    def update_tracking(self, frame: Any) -> Optional[Tuple[int, int, int, int]]:
        if not self.is_tracking:
            return None
        
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = tuple(map(int, bbox))
            return self.bbox
        else:
            return None

    def handle_lost_tracking(self, frame: Any, detector: ObjectDetector) -> Optional[Tuple[int, int, int, int]]:
        detections = detector.detect_objects(frame)
        if detections:
            new_detection = detections[0]
            bbox = tuple(map(int, new_detection.box))
            self.start_tracking(frame, bbox)
            return bbox
        return None
