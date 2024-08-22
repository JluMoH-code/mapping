from typing import List

class Detection:
    def __init__(self, box: List[float], class_id: int, confidence: float):
        self.box = box              # (x, y, w, h)
        self.class_id = class_id
        self.confidence = confidence
