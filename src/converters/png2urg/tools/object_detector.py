import numpy as np
import logging
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image


class ResizeAndPad:
    def __init__(self, output_size, fill=0):
        self.output_size = output_size
        self.fill = fill

    def __call__(self, image, target=None):
        w, h = image.size
        scale = self.output_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = F.resize(image, (new_h, new_w))

        new_image = Image.new('RGB', (self.output_size, self.output_size), (self.fill, self.fill, self.fill))
        pad_x = (self.output_size - new_w) // 2
        pad_y = (self.output_size - new_h) // 2
        new_image.paste(image, (pad_x, pad_y))

        if target and "boxes" in target:
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
            target["boxes"] = boxes

        return new_image, target
    

class ToTensor:
    def __call__(self, image, target=None):
        return F.to_tensor(image), target
    

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    

def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class ObjectDetector:
    def __init__(self, model_path: str, num_classes: int, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = get_object_detection_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transforms = Compose([
            ResizeAndPad(800),
            ToTensor()
        ])

    def predict(self, image: np.ndarray, confidence_threshold: float = 0.5) -> dict:
        logging.info(f"Starting prediction with confidence threshold: {confidence_threshold}")

        image = Image.fromarray(image)
        original_w, original_h = image.size
        
        image_tensor, _ = self.transforms(image)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            prediction = self.model([image_tensor])[0]

        mask = prediction['scores'] > confidence_threshold
        pred_boxes = prediction['boxes'][mask].cpu().numpy()
        labels = prediction['labels'][mask].cpu().numpy()
        scores = prediction['scores'][mask].cpu().numpy()

        scale = 800 / max(original_w, original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        pad_x = (800 - new_w) // 2
        pad_y = (800 - new_h) // 2

        original_boxes = pred_boxes.copy()
        original_boxes[:, [0, 2]] = (original_boxes[:, [0, 2]] - pad_x) / scale
        original_boxes[:, [1, 3]] = (original_boxes[:, [1, 3]] - pad_y) / scale

        original_boxes[:, 0] = np.clip(original_boxes[:, 0], 0, original_w)
        original_boxes[:, 1] = np.clip(original_boxes[:, 1], 0, original_h)
        original_boxes[:, 2] = np.clip(original_boxes[:, 2], 0, original_w)
        original_boxes[:, 3] = np.clip(original_boxes[:, 3], 0, original_h)

        logging.info(f"Prediction completed with {len(original_boxes)} boxes after thresholding")

        return {
            'boxes': original_boxes,
            'labels': labels,
            'scores': scores
        }