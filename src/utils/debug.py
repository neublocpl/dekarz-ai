import numpy as np
import cv2

from src.config import DEBUG_FILES_DIR


_current_job_id = ""


def set_current_job_id(job_id: str):
    global _current_job_id
    _current_job_id = job_id


def save_debug_image_file(image: np.ndarray, suffix: str):
    if not _current_job_id:
        return

    file_path = DEBUG_FILES_DIR / f"{_current_job_id}_{suffix}.png"
    cv2.imwrite(str(file_path), image)
    
    