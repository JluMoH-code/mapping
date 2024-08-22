import cv2
from typing import Optional, Tuple, Any, Union
from ObjectDetector import ObjectDetector
from DisplayUtils import DisplayUtils

class ObjectTracker:
    def __init__(self, tracker_type: str = 'KCF'):
        self.tracker_type = tracker_type
        self.create_tracker(tracker_type)
        self.is_tracking = False
        self.bbox = None            # (x, y, width, height)

    def create_tracker(self, tracker_type: str) -> None:
        if tracker_type == 'MIL':
            self.tracker = cv2.legacy.TrackerMIL.create()
        elif tracker_type == 'BOOSTING':
            self.tracker = cv2.legacy.TrackerBoosting.create()
        elif tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.legacy.TrackerMedianFlow.create()
        elif tracker_type == 'TLD':
            self.tracker = cv2.legacy.TrackerTLD.create()
        elif tracker_type == 'KCF':
            self.tracker = cv2.legacy.TrackerKCF.create()
        elif tracker_type == 'MOSSE':
            self.tracker = cv2.legacy.TrackerMOSSE.create()
        elif tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN.create()
        elif tracker_type == 'DASIAMRPN':
            self.tracker = cv2.TrackerDaSiamRPN.create()
        elif tracker_type == 'NANO':
            self.tracker = cv2.TrackerNano.create()
        elif tracker_type == 'VIT':
            self.tracker = cv2.TrackerVit.create()
        
        else:
            raise ValueError(f"Неизвестный тип трекера: {tracker_type}")

    def yolobbox2bbox(self, bbox):
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    
    def bbox2yolobbox(self, bbox):
        return [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]

    def start_tracking(self, frame: Any, bbox: Tuple[int, int, int, int], min_size: int = 80) -> None:        
        yolobbox = DisplayUtils.calculate_centered_area(bbox, frame.shape, min_size=min_size)
        self.bbox = self.yolobbox2bbox(yolobbox)
        self.is_tracking = self.tracker.init(frame, self.bbox)

    def reinitialize_tracker(self, frame: Any, bbox: Tuple[int, int, int, int], min_size: int = 80) -> None:
        self.create_tracker(self.tracker_type)
        self.start_tracking(frame, bbox, min_size)

    def is_update_tracking(self, frame: Any) -> bool:
        if not self.is_tracking:
            return self.is_tracking
        
        success, self.bbox = self.tracker.update(frame)
        return success

    def handle_lost_tracking(self, frame: Any, detector: ObjectDetector) -> Optional[Tuple[int, int, int, int]]:
        detections = detector.detect_objects(frame)
        if detections:
            new_detection = detections[0]
            bbox = tuple(map(int, new_detection.box))
            self.start_tracking(frame, bbox)
            return bbox
        return None

    def tracking(self, frame: Any, show: bool = False) -> bool:
        self.is_tracking = self.is_update_tracking(frame)
        
        if self.bbox and show:
            tracking_frame = frame.copy()
            DisplayUtils.draw_detection_box(tracking_frame, self.bbox2yolobbox(self.bbox))
            DisplayUtils.show_frame(tracking_frame, window_name="Tracking")
            
            centered_area = DisplayUtils.calculate_centered_area(self.bbox2yolobbox(self.bbox), frame.shape, min_size=150)
            x1, y1, x2, y2 = centered_area
            cropped_frame = frame[y1:y2, x1:x2]
            DisplayUtils.show_frame(cropped_frame, window_name="Centered Object")
        
        return self.is_tracking