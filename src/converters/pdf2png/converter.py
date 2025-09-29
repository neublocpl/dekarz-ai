import io
import cv2
import numpy as np
import logging
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

from src.utils import debug


class Pdf2PngConverter:
    def run(self, input_data: bytes) -> np.ndarray | None:
        if not input_data:
            logging.error("No input data provided for PDF to PNG conversion.")
            return None
        
        try:
            images = convert_from_bytes(input_data)

            if not images:
                logging.error("No images were extracted from the PDF data.")
                return None
            
            image = images[0]
            rotated_image = image.rotate(90, expand=True)
            
            image_array = np.array(rotated_image)
            logging.info(f"Converted PDF to PNG with shape: {image_array.shape}")
            
            debug_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            debug.save_debug_image_file(debug_image, "01")
            
            return image_array
        
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            logging.error(f"Error during PDF to PNG conversion: {e}")
            return None