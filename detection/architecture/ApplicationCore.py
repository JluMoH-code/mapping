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
        self.display = DisplayUtils()
        self.input_handler = InputHandler()

        self.is_tracking = False
        self.detections = None
        
    def update_and_show_selected_object(self, detections, frame, context_scale=1.5, min_size=150):
        new_selected_object = self.selector.select_object(detections, frame)

        if new_selected_object:
            self.selector.selected_object = new_selected_object

        if self.selector.selected_object:
            self.display.show_selected_object(frame, self.selector.selected_object, context_scale, min_size)

    def run(self):
        try:
            while True:
                # 1. Получение следующего кадра
                self.frame = self.frame_capture.get_frame()
                resized_frame = self.display.resize_frame(self.frame, 640, 384)

                # 2. Обнаружение объектов
                self.detections = self.detector.detect_objects(resized_frame)

                # 3. Отрисовка обнаруженных объектов на кадре
                frame_with_detections = self.display.draw_detections(resized_frame, self.detections)

                # 4. Выбор объекта (если требуется) и его отображение
                self.update_and_show_selected_object(self.detections, resized_frame)

                # 4. Отслеживание объекта
                # if self.is_tracking:
                #     self.track_object(frame)

                # 5. Отображение или запись результата
                self.display.show_frame(frame_with_detections)
                
                # Проверка на выход
                if self.input_handler.should_exit():
                    break

        except KeyboardInterrupt:
            print("Приложение остановлено пользователем.")
        finally:
            self.frame_capture.release()
            self.display.close_windows()
