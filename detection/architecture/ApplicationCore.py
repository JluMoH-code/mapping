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

    def draw_detections(self, frame, detections):
        """
        Отрисовывает обнаруженные объекты на кадре.
        """
        for detection in detections:
            # Распаковываем координаты
            x_min, y_min, x_max, y_max = map(int, detection.box)

            # Отрисовываем прямоугольник вокруг обнаруженного объекта
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Отрисовываем класс и уверенность
            label = f"Class: {detection.class_id}, Confidence: {detection.confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def run(self):
        try:
            cv2.namedWindow("Frame")
            while True:
                # 1. Получение следующего кадра
                frame = self.get_frame()

                detections = self.detector.detect_objects(frame)

                # 3. Отрисовка обнаруженных объектов на кадре
                frame_with_detections = self.draw_detections(frame, detections)

                # 3. Если объект еще не выбран, предложить выбрать
                if not self.is_tracking:
                    selected_object = self.selector.select_object(detections, frame_with_detections)
                if selected_object:
                    self.selected_object = selected_object
                    self.is_tracking = True

                    # Показ выбранного объекта в отдельном окне
                    selected_box = self.selected_object.box
                    selected_frame = frame[int(selected_box[1]):int(selected_box[3]), 
                                           int(selected_box[0]):int(selected_box[2])]
                    cv2.imshow('Selected Object', selected_frame)

                # 4. Отслеживание объекта
                # if self.is_tracking:
                #     self.track_object(frame)

                # 5. Отображение или запись результата
                cv2.imshow('Frame', frame_with_detections)
                cv2.waitKey(1)

        except KeyboardInterrupt:
            print("Приложение остановлено пользователем.")
        finally:
            self.frame_capture.release()
