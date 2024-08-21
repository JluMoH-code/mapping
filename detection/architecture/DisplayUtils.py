import cv2
import numpy as np
from typing import List, Optional, Any, Tuple
from ObjectDetector import Detection

class DisplayUtils:
    def __init__(self, window_name: str = "Frame"):
        self.selected_frame = None
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def draw_detection_box(self, frame: Any, bbox: List[float]) -> None:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    def draw_detections(self, frame: Any, detections: List[Detection]) -> Any:
        frame_copy = np.copy(frame)

        for detection in detections:
            box = detection.box
            class_id = detection.class_id
            confidence = detection.confidence

            self.draw_detection_box(frame_copy, box)

            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            cv2.putText(frame_copy, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame_copy

    def calculate_centered_area(self, box: List[int], frame_shape: Tuple[int, int], context_scale: float, min_size: int = 150) -> Tuple[int, int, int, int]:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        width = max((box[2] - box[0]) * context_scale, min_size)
        height = max((box[3] - box[1]) * context_scale, min_size)

        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame_shape[1])
        y2 = min(y2, frame_shape[0])

        if (x2 - x1) < min_size:
            if x1 == 0:
                x2 = min_size
            elif x2 == frame_shape[1]:
                x1 = max(frame_shape[1] - min_size, 0)
            else:
                center_x = (x1 + x2) / 2
                x1 = int(center_x - min_size / 2)
                x2 = x1 + min_size

        if (y2 - y1) < min_size:
            if y1 == 0:
                y2 = min_size
            elif y2 == frame_shape[0]:
                y1 = max(frame_shape[0] - min_size, 0)
            else:
                center_y = (y1 + y2) / 2
                y1 = int(center_y - min_size / 2)
                y2 = y1 + min_size

        return x1, y1, x2, y2

    def ensure_minimum_size(self, frame: Any, min_size_window: int) -> Any:
        if frame.shape[0] < min_size_window or frame.shape[1] < min_size_window:
            return self.resize_frame(frame, min_size_window, min_size_window)
        return frame

    def show_selected_object(self, frame: Any, selected_object: Optional['Detection'], context_scale: float = 1.5, min_size_area: int = 150, min_size_window: int = 150) -> None:
        if selected_object:
            x1, y1, x2, y2 = self.calculate_centered_area(selected_object.box, frame.shape, context_scale, min_size_area)
            self.selected_frame = frame[y1:y2, x1:x2]
            self.selected_frame = self.ensure_minimum_size(self.selected_frame, min_size_window)
            cv2.imshow('Selected Object', self.selected_frame)
            return self.selected_frame

    def show_frame(self, frame: Any, window_name: str = "Frame") -> None:
        cv2.imshow(window_name, frame)

    def close_windows(self) -> None:
        cv2.destroyAllWindows()

    def resize_frame(self, frame: Any, width: int, height: int) -> Any:
        return cv2.resize(frame, (width, height))

    def resize_window(self, width: int, height: int) -> None:
        cv2.resizeWindow(self.window_name, width, height)