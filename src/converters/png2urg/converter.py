import cv2
import logging
import numpy as np

from src import config
from src.utils import debug
from src.converters import AbstractUrgConverter
from src.schemas import URG
from src.converters.png2urg.tools.object_detector import ObjectDetector


class Png2UrgConverter(AbstractUrgConverter):
    def run(self, input_data: np.ndarray) -> URG:
        unnecessary_elements_detector = ObjectDetector(model_path=config.UNNECESESARY_ELEMENTS_MODEL_PATH, num_classes=5)
        predictions = unnecessary_elements_detector.predict(input_data, confidence_threshold=0.5)

        # --- DEBUG ------
        debug_image = input_data.copy()
        for box in predictions['boxes']:
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(debug_image, pt1, pt2, (255, 0, 0), 2)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
        debug.save_debug_image_file(debug_image, "02_1")
        # ----------------

        cleaned_image = input_data.copy()
        img_h, img_w, _ = cleaned_image.shape
        padding_factor = 0.05  # 10% padding
        for box in predictions['boxes']:
            x1, y1, x2, y2 = box
            
            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = box_w * padding_factor
            pad_h = box_h * padding_factor
            
            new_x1 = x1 - pad_w
            new_y1 = y1 - pad_h
            new_x2 = x2 + pad_w
            new_y2 = y2 + pad_h

            pt1 = (int(max(0, new_x1)), int(max(0, new_y1)))
            pt2 = (int(min(img_w, new_x2)), int(min(img_h, new_y2)))
            cv2.rectangle(cleaned_image, pt1, pt2, (255, 255, 255), -1)

        # --- DEBUG ------
        debug_image = cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2BGR)
        debug.save_debug_image_file(debug_image, "02_2")
        # ----------------

        logging.info(f"Converted PNG to URG format")
        return URG()