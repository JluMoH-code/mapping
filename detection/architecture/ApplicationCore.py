import FrameCapture
import ObjectDetector
import ObjectTracker
import ObjectSelector
from DisplayUtils import DisplayUtils
from InputHandler import InputHandler

class ApplicationCore:
    def __init__(self, frame_capture: FrameCapture, detector: ObjectDetector, selector: ObjectSelector, tracker: ObjectTracker):
        self.frame_capture = frame_capture
        self.detector = detector
        self.selector = selector
        self.tracker = tracker
        self.input_handler = InputHandler()
        
    def update_and_show_selected_object(self, detections, frame, context_scale=1.5, min_size_area=150, min_size_window=150):
        new_selected_object = self.selector.select_object(detections)

        if new_selected_object:
            self.selector.selected_object = new_selected_object
            bbox = tuple(map(int, new_selected_object.box))
            self.tracker.reinitialize_tracker(frame, bbox, context_scale=context_scale, min_size=min_size_area)

    def run(self):
        try:
            while True:
                self.frame = self.frame_capture.get_frame()
                resized_frame = DisplayUtils.resize_frame(self.frame, 640, 384)
                DisplayUtils.show_frame(resized_frame)
                
                self.detector.detect_objects(resized_frame, show = True)
                
                self.update_and_show_selected_object(self.detector.detections, resized_frame, min_size_area=80)

                self.tracker.tracking(resized_frame, show=True)
                        
                if self.input_handler.should_exit():
                    break

        except KeyboardInterrupt:
            print("Приложение остановлено пользователем.")
        finally:
            self.frame_capture.release()
            DisplayUtils.close_windows()
