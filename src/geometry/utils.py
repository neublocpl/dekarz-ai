import os
import pickle

import cv2
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from src.geometry.objects import Interval


def load_file(file_path, **kwargs):
    if file_path.lower().endswith(".pdf"):
        return convert_to_image(file_path=file_path, **kwargs)
    else:
        load_image(file_path)


def load_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"File not found: {file_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, gray


def convert_to_image(file_path, dpi=300, page_number: int = 0):
    doc = fitz.open(file_path)
    page = doc[page_number]
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def show_images(images, show=True, figsize=(30, 30), labels=None):
    N = len(images)
    labels = labels or [None] * N
    ig, axs = plt.subplots(1, N, figsize=figsize)
    for i, (img, label) in enumerate(zip(images, labels)):
        if len(img.shape) == 2:
            axs[i].imshow(img, cmap="gray")
        else:
            axs[i].imshow(img)
        if label:
            axs[i].set_title(label)
        axs[i].axis("off")  # Hide the axes
    plt.tight_layout()
    if show:
        plt.show()


def draw_lines(image, intervals: list[Interval], type_mapping=Config.COLOR_MAP):
    img_viz = image.copy()
    for interval in intervals:
        p1, p2 = interval.endpoints
        # Draw a green line over the detected segment
        cv2.line(img_viz, p1, p2, type_mapping.get(interval.classification), 2)
        # Add a text label with the classification info
        # label = f"{interval.classification} (T:{interval.thickness})"
        # text_pos = (p1[0] + 5, p1[1] - 8)
        # cv2.putText(img_viz, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    plt.figure(figsize=(25, 25))
    plt.imshow(img_viz)
    plt.show()


def dump_sessions_to_pickle(step, step_name):
    """
    Save ALL_SESSIONS dictionary to a pickle file.

    Args:
        file_path: path to save the pickle file.
    """
    file_path = f"{step_name}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(step, f)


def load_sessions_from_pickle(step_name):
    """
    Load ALL_SESSIONS dictionary from a pickle file.

    Args:
        file_path: path to load the pickle file from.

    Returns:
        dict: The loaded ALL_SESSIONS dictionary.
    """
    file_path = f"{step_name}.pkl"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)
