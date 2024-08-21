import cv2

class InputHandler:
    def __init__(self, exit_key: int = 27):
        self.exit_key = exit_key

    def should_exit(self, wait_time: int = 1) -> bool:
        return cv2.waitKey(wait_time) == self.exit_key
