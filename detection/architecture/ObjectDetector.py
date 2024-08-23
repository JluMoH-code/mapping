from Detection import Detection
from DisplayUtils import DisplayUtils
from typing import List, Any
from ultralytics import YOLO
import torchvision.models.detection as detection
import torchvision.transforms as transforms
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

class ObjectDetector:
    def load_model(self, model_path: str) -> Any:
        raise NotImplementedError

    def detect_objects(self, image: Any) -> List[Detection]:
        raise NotImplementedError
    
    def check_model(self) -> None:
        raise NotImplementedError
    
class YOLODetector(ObjectDetector):
    def __init__(self, model_path: str = "yolov8n", confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = self.load_model(model_path)
        self.detections = []
        
    def get_center_bbox(self, bbox):
        return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

    def get_detections(self):
        detections_output = []
        for detection in self.detections:
            detections_output.append({
                "class_id": detection.class_id,
                "bbox": detection.box,
                "confidence": detection.confidence,
                "center_bbox": self.get_center_bbox(detection.box),
            })
        return detections_output
        
    def load_model(self, model_path: str) -> YOLO:
        return YOLO(model_path)

    def detect_objects(self, frame: Any, verbose: bool = False, show: bool = True) -> List[Detection]:
        results = self.model(frame, verbose=verbose)
        self.detections = []
        for result in results[0].boxes:
            confidence = result.conf[0].item()
            if confidence >= self.confidence_threshold:
                box = result.xyxy[0].tolist()
                class_id = int(result.cls[0].item())
                confidence = float(confidence)
                self.detections.append(Detection(box, class_id, confidence))
                
        if len(self.detections) and show:
            detection_frame = frame.copy()
            detection_frame = DisplayUtils.draw_detections(detection_frame, self.detections)
            DisplayUtils.show_frame(detection_frame, window_name="Detection") 
                
        return self.detections
    
    def check_model(self) -> None:
        if self.model is None:
            raise Exception("Модель не загружена правильно.")
        print("Модель загружена успешно.")
        
class SSDObjectDetector(ObjectDetector):
    def __init__(self, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = self.load_model()
    
    def load_model(self) -> torch.nn.Module:
        model = detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        model.eval()
        return model
    
    def detect_objects(self, image: Any, show: bool = False) -> List[Detection]:
        transform = transforms.ToTensor()
        image = transform(image).unsqueeze(0)
        results = self.model(image)[0]
        detections = []
        for i, score in enumerate(results['scores']):
            if score > self.confidence_threshold:
                box = results['boxes'][i].tolist()
                class_id = int(results['labels'][i].item())
                confidence = float(score.item())
                detections.append(Detection(box, class_id, confidence))
        return detections
    
class FasterRCNNObjectDetector(ObjectDetector):
    def __init__(self, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = self.load_model()
        
    def load_model(self) -> torch.nn.Module:
        model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model
    
    def detect_objects(self, image: Any, show: bool = False) -> List[Detection]:
        transform = transforms.ToTensor()
        image = transform(image).unsqueeze(0)
        results = self.model(image)[0]
        detections = []
        for i, score in enumerate(results['scores']):
            if score > self.confidence_threshold:
                box = results['boxes'][i].tolist()
                class_id = int(results['labels'][i].item())
                confidence = float(score.item())
                detections.append(Detection(box, class_id, confidence))
        return detections
    
class Detectron2ObjectDetector(ObjectDetector):
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path: str) -> DefaultPredictor:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        cfg.MODEL.DEVICE = 'cpu'
        return DefaultPredictor(cfg)
    
    def detect_objects(self, image: Any, show: bool = False) -> List[Detection]:
        outputs = self.model(image)
        detections = []
        instances = outputs["instances"].to("cpu")
        for i in range(len(instances)):
            box = instances.pred_boxes[i].tensor.numpy().tolist()
            class_id = int(instances.pred_classes[i].item())
            confidence = float(instances.scores[i].item())
            detections.append(Detection(box, class_id, confidence))
        return detections