import FrameCapture
import ObjectDetector
import ObjectTracker
import ObjectSelector
from DisplayUtils import DisplayUtils
from InputHandler import InputHandler
import cv2

class ApplicationCore:
    def __init__(self, frame_capture: FrameCapture, detector: ObjectDetector, selector: ObjectSelector, tracker: ObjectTracker):
        self.frame_capture = frame_capture
        self.detector = detector
        self.selector = selector
        self.tracker = tracker
        self.display = DisplayUtils()
        self.input_handler = InputHandler()

        self.is_tracking = False
        self.detections = None
        
    def update_and_show_selected_object(self, detections, frame, context_scale=1.5, min_size_area=150, min_size_window=150):
        new_selected_object = self.selector.select_object(detections, frame)

        if new_selected_object:
            self.selector.selected_object = new_selected_object
            bbox = tuple(map(int, new_selected_object.box))
            self.tracker.start_tracking(frame, bbox)
            self.is_tracking = True

        if self.selector.selected_object:
            self.display.show_selected_object(frame, self.selector.selected_object, context_scale, min_size_area, min_size_window)

    def run(self):
        try:
            while True:
                # 1. Получение следующего кадра
                self.frame = self.frame_capture.get_frame()
                resized_frame = self.display.resize_frame(self.frame, 640, 384)
                self.display.show_frame(resized_frame)

                if self.is_tracking:
                # Режим трекинга
                    success, bbox = self.tracker.update(resized_frame)
                    
                    if success:
                        # Если трекинг успешен, отображаем объект
                        tracking_frame = resized_frame.copy()
                        self.display.draw_detection_box(tracking_frame, bbox)
                        self.display.show_frame(tracking_frame, window_name="Tracking")
                    else:
                        # Если трекер потерял объект, завершаем трекинг
                        print("Трекер потерял объект.")
                        self.is_tracking = False
                else:
                    # Режим детекции и выбора
                    self.detections = self.detector.detect_objects(resized_frame)
                    detection_frame = resized_frame.copy()
                    detection_frame = self.display.draw_detections(detection_frame, self.detections)
                    self.display.show_frame(detection_frame, window_name="Detection")

                    new_selected_object = self.selector.select_object(self.detections, resized_frame)
                    if new_selected_object:
                        self.selector.selected_object = new_selected_object
                        bbox = tuple(map(int, new_selected_object.box))
                        # Инициализация трекера с выбранным объектом
                        self.tracker = cv2.legacy.TrackerMOSSE_create()  # Например, KCF трекер
                        self.tracker.init(resized_frame, bbox)
                        self.is_tracking = True
                        print("Трекинг начат.")
                        
                # Проверка на выход
                if self.input_handler.should_exit():
                    break

        except KeyboardInterrupt:
            print("Приложение остановлено пользователем.")
        finally:
            self.frame_capture.release()
            self.display.close_windows()
