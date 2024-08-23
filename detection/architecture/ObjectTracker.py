import cv2
from typing import Optional, Tuple, Any
from ObjectDetector import ObjectDetector
from DisplayUtils import DisplayUtils

class ObjectTracker:
    TRACKERS = {
        'MIL': cv2.legacy.TrackerMIL,
        'BOOSTING': cv2.legacy.TrackerBoosting,
        'MEDIANFLOW': cv2.legacy.TrackerMedianFlow,
        'TLD': cv2.legacy.TrackerTLD,
        'KCF': cv2.legacy.TrackerKCF,
        'MOSSE': cv2.legacy.TrackerMOSSE,
        'GOTURN': cv2.TrackerGOTURN,
        'DASIAMRPN': cv2.TrackerDaSiamRPN,
        'NANO': cv2.TrackerNano,
        'VIT': cv2.TrackerVit
    }
    
    def __init__(self, tracker_type: str = 'KCF'):
        self.tracker_type = tracker_type
        self.is_tracking = False
        self.bbox = None            # (x, y, width, height)

    def create_tracker(self, tracker_type: str) -> None:
        if tracker_type not in self.TRACKERS:
            raise ValueError(f"Неизвестный тип трекера: {tracker_type}")
        
        self.tracker = self.TRACKERS[tracker_type].create()

    def yolobbox2bbox(self, bbox):
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    
    def bbox2yolobbox(self, bbox):
        return [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]

    def start_tracking(self, frame: Any, bbox: Tuple[int, int, int, int], context_scale: float = 1.5, min_size: int = 80) -> None:        
        yolobbox = DisplayUtils.calculate_centered_area(bbox, frame.shape, context_scale=context_scale, min_size=min_size)
        self.bbox = self.yolobbox2bbox(yolobbox)
        self.is_tracking = self.tracker.init(frame, self.bbox)

    def reinitialize_tracker(self, frame: Any, bbox: Tuple[int, int, int, int], context_scale: float = 1.5, min_size: int = 80) -> None:
        self.create_tracker(self.tracker_type)
        self.start_tracking(frame, bbox, context_scale=context_scale, min_size=min_size)

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