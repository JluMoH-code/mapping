import FrameCapture
import ObjectDetector
import ObjectTracker
import ObjectSelector
from DisplayUtils import DisplayUtils
from InputHandler import InputHandler
import json
import numpy as np
from time import time

class ApplicationCore:
    def __init__(self, frame_capture: FrameCapture, detector: ObjectDetector, selector: ObjectSelector, tracker: ObjectTracker):
        self.frame_capture = frame_capture
        self.detector = detector
        self.selector = selector
        self.tracker = tracker
        self.input_handler = InputHandler()
        self.json_data = []
        self.reload_tracker_by_detector_interval_sec = 1
        self.frame_number = 0
        self.frame_time = time()
        self.time_last_update = time()
        self.total_time = 0
        
    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def update_tracker_by_detector(self, detections, frame, context_scale=1.5, min_size_area=150, distance_threshold=25.0):
        self.detector.detect_objects(frame)
        min_distance = float('inf')
        matched_detected = None
        
        for detection in detections:
            detected_center = self.detector.get_center_bbox(detection.box)
            tracked_center = self.tracker.get_center_bbox()
            
            distance = self.euclidean_distance(detected_center, tracked_center)
            
            print(detected_center, tracked_center, distance)
            
            if distance < min_distance and distance <= distance_threshold:
                min_distance = distance
                matched_detected = detection.box
                
        if matched_detected and min_distance >= distance_threshold / 5:
            print(f"Трекер скорректирован на точку: {matched_detected} (расстояние: {min_distance}px)")
            self.tracker.reinitialize_tracker(frame, matched_detected, context_scale=context_scale, min_size=min_size_area)
        return matched_detected

    def forming_output_data(self, frame_number):
        data = {
            "frame_number": frame_number,
            "is_tracking": self.tracker.is_tracking,
            "tracked_bbox": self.tracker.get_bbox(),
            "center_tracked_bbox": self.tracker.get_center_bbox(),
            "detected_bboxes": self.detector.get_detections(),
        }
        self.json_data.append(data)

    def save_data(self):
        output_path = "C:/Users/User/Documents/python/mapping/detection/results.json"
        with open(output_path, "w") as f:
            json.dump(self.json_data, f, indent=4)           

    def calculate_fps(self):
        fps = 1 / (time() - self.frame_time)
        self.frame_time = time()
        return fps

    def run(self):
        try:
            while True:
                self.frame = self.frame_capture.get_frame()
                resized_frame = DisplayUtils.resize_frame(self.frame, 640, 384)
                DisplayUtils.show_frame(resized_frame)
                
                DisplayUtils.check_click(self.selector, self.detector, self.tracker, resized_frame, show=True, window_name="Frame", min_size_area=80)
                
                if time() - self.time_last_update >= self.reload_tracker_by_detector_interval_sec:
                    self.time_last_update = time()
                    self.update_tracker_by_detector(self.detector.detections, resized_frame, min_size_area=80)

                self.tracker.tracking(resized_frame, show=True)
                
                self.forming_output_data(self.frame_number)
                
                self.frame_number += 1
                
                fps = self.calculate_fps()
                print(f"Средний FPS: {fps:.2f}")
                        
                if self.input_handler.should_exit():
                    break

        except KeyboardInterrupt:
            print("Приложение остановлено пользователем.")
        finally:
            self.frame_capture.release()
            DisplayUtils.close_windows()
            self.save_data()
