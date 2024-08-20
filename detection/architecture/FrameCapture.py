import cv2
import time

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
except ImportError:
    PiCamera = None
    PiRGBArray = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

class FrameCapture:
    def get_frame(self):
        raise NotImplementedError
    
    def release(self):
        raise NotImplementedError
    
    def is_opened(self):
        raise NotImplementedError
    
    def configure(self, **kwargs):
        raise NotImplementedError

class OpenCVFrameCapture(FrameCapture):
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.is_opened():
            raise Exception(f"Не удалось открыть видеофайл по пути: {video_path}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Не удалось получить кадр из видеопотока")
        return frame
    
    def release(self):
        self.cap.release()
        
    def is_opened(self):
        return self.cap.isOpened()
    
class ImageFrameCapture(FrameCapture):
    def __init__(self, file_path):
        self.file_path = file_path
        self.image = cv2.imread(file_path)
        if self.image is None:
            raise Exception(f"Не удалось загрузить изображение по пути: {file_path}")

    def get_frame(self):
        return self.image
    
    def release(self):
        pass
    
    def is_opened(self):
        return self.image is not None
 
if PiCamera:       
    class PiCameraCapture(FrameCapture):
        def __init__(self, resolution=(640, 480), framerate=32):
            self.camera = PiCamera()
            self.configure(resolution, framerate)
            self.raw_capture = PiRGBArray(self.camera, size=resolution)
            time.sleep(0.1)

        def get_frame(self):
            try:
                self.camera.capture(self.raw_capture, format="bgr")
                return self.raw_capture.array
            except Exception as e:
                raise Exception(f"Ошибка при захвате кадра: {e}")

        def release(self):
            self.camera.close()

        def is_opened(self):
            return True
        
        def configure(self, resolution, framerate):
            self.camera.resolution = resolution
            self.camera.framerate = framerate
    
if Picamera2:    
    class PiCamera2Capture(FrameCapture):
        def __init__(self):
            self.camera = Picamera2()
            self.configure()
            self.camera.start()

        def get_frame(self):
            try:
                return self.camera.capture_array()
            except Exception as e:
                raise Exception(f"Ошибка при захвате кадра: {e}")

        def release(self):
            self.camera.close()

        def is_opened(self):
            return True
        
        def configure(self):
            self.camera.configure(self.camera.create_still_configuration())