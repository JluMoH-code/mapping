import FrameCapture
import ObjectDetector
import ObjectTracker
import ObjectSelector
import cv2

class ApplicationCore:
    def __init__(self, frame_capture: FrameCapture, detector: ObjectDetector, selector: ObjectSelector, tracker: ObjectTracker):
        self.frame_capture = frame_capture
        self.detector = detector
        self.selector = selector
        self.tracker = tracker

        self.selected_object = None
        self.is_tracking = False
        
    def get_frame(self):
        return self.frame_capture.get_frame()

    def detect_objects(self, image):
        return self.detector.detect_objects(image)
    
    def select_object(self, objects, method):
        self.selected_object = self.selector.select_object(objects, method)
        return self.selected_object

    def track_object(self, image):
        if self.selected_object:
            new_coordinates = self.tracker.track_object(image, self.selected_object)
            self.selected_object = new_coordinates
            return new_coordinates
        return None

    def run(self):
        try:
            while True:
                # 1. Получение следующего кадра
                frame = self.get_frame()

                # 2. Обнаружение объектов
                # objects = self.detect_objects(frame)

                # 3. Если объект еще не выбран, предложить выбрать
                # if not self.is_tracking:
                #     self.selected_object = self.select_object(objects, method='click')
                #     self.is_tracking = True

                # 4. Отслеживание объекта
                # if self.is_tracking:
                #     self.track_object(frame)

                # 5. Отображение или запись результата
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)

        except KeyboardInterrupt:
            print("Приложение остановлено пользователем.")
        finally:
            self.frame_capture.release()
