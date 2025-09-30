# import easyocr
# import cv2
# import os
# import numpy as np
# from uuid import uuid4
# import matplotlib.pyplot as plt
# from src.geometry.utils import convert_to_image

# # from src.utils.schema import Direction


# OCR_READER = easyocr.Reader(['en'], gpu=False)

# # TEMPLATE_DIR = "/app/src/utils/templates/"
# # ARROW_TEMPLATE_FILES = {
# #     Direction.NORTH: "n_arrow.png",
# #     Direction.SOUTH: "s_arrow.png",
# #     Direction.EAST: "e_arrow.png",
# #     Direction.WEST: "w_arrow.png",
# # }

# # LOADED_ARROW_TEMPLATES = {}
# # for direction, filename in ARROW_TEMPLATE_FILES.items():
# #     path = os.path.join(TEMPLATE_DIR, filename)
# #     if os.path.exists(path):
# #         template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# #         if template is not None:
# #             LOADED_ARROW_TEMPLATES[direction] = template


# # def extract_arrow_direction(image: np.ndarray) -> Direction | None:
# #     if image is None or image.size == 0:
# #         return None
    
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     mser = cv2.MSER_create()
# #     regions, _ = mser.detectRegions(gray)

# #     # Remove values
# #     for region in regions:
# #         x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
# #         if 5 < w < 50 and 5 < h < 50:
# #             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255, 255), -1)

# #     # Remove degree symbol
# #     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=10, minRadius=2, maxRadius=5)
# #     if circles is not None:
# #         circles = np.uint16(np.around(circles))
# #         for i in circles[0, :]:
# #             padded_radius = i[2] + 3
# #             cv2.circle(image, (i[0], i[1]), (i[2]+3), (255, 255, 255, 255), -1)

# #     # Remove unnecessery lines
# #     height, width = gray.shape
# #     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=int(min(height, width) * 0.97), maxLineGap=5)
# #     if lines is not None:
# #         line_image = np.copy(image)
# #         for line in lines:
# #             x1, y1, x2, y2 = line[0]
# #             if abs(y1 - y2) < 10 and abs(x1 - x2) >= width * 0.97:
# #                 cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255, 255), 2)
# #             if abs(x1 - x2) < 10 and abs(y1 - y2) >= height * 0.97:
# #                 cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255, 255), 2)

# #     # Find our arrow
# #     cleaned_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     cleaned_gray = cv2.GaussianBlur(cleaned_gray, (3, 3), 0)
# #     edges = cv2.Canny(cleaned_gray, 50, 150, apertureSize=3)
# #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=25, maxLineGap=7)

# #     longest_line = None
# #     max_length = 0

# #     if lines is not None:
# #         is_horizontal_img = width > height
# #         for line in lines:
# #             x1, y1, x2, y2 = line[0]
# #             length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# #             if is_horizontal_img and abs(y1 - y2) < 10:
# #                 if length > max_length:
# #                     max_length = length
# #                     longest_line = line
# #             elif not is_horizontal_img and abs(x1 - x2) < 10:
# #                 if length > max_length:
# #                     max_length = length
# #                     longest_line = line

# #     PADDING_H_X = 0
# #     PADDING_H_Y = 20
# #     PADDING_V_X = 20
# #     PADDING_V_Y = 0

# #     x1, y1, x2, y2 = longest_line[0]

# #     # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255, 255), 2)
# #     # cv2.imwrite(f"/app/data/output/roi/{uuid4()}.png", image)

# #     if is_horizontal_img:
# #         start_x = min(x1, x2) - PADDING_H_X
# #         end_x = max(x1, x2) + PADDING_H_X
# #         center_y = int((y1 + y2) / 2)
# #         start_y = center_y - PADDING_H_Y
# #         end_y = center_y + PADDING_H_Y
# #     else:
# #         start_y = min(y1, y2) - PADDING_V_Y
# #         end_y = max(y1, y2) + PADDING_V_Y
# #         center_x = int((x1 + x2) / 2)
# #         start_x = max(0, center_x - PADDING_V_X)
# #         end_x = min(width, center_x + PADDING_V_X)

# #     start_x, end_x, start_y, end_y = int(start_x), int(end_x), int(start_y), int(end_y)
# #     cropped_image = image[start_y:end_y, start_x:end_x]

# #     gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# #     if is_horizontal_img:
# #         _, thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
# #         _, w = thresh.shape
# #         mid_point = w // 2

# #         left_half_sum = np.sum(thresh[:, :mid_point])
# #         right_half_sum = np.sum(thresh[:, mid_point:])

# #         if left_half_sum > right_half_sum:
# #             return Direction.WEST
# #         else:
# #             return Direction.EAST
# #     else:
# #         _, thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
# #         h, _ = thresh.shape
# #         mid_point = h // 2

# #         top_half_sum = np.sum(thresh[0:mid_point, :])
# #         bottom_half_sum = np.sum(thresh[mid_point:h, :])

# #         if top_half_sum > bottom_half_sum:
# #             return Direction.NORTH
# #         else:
# #             return Direction.SOUTH 

# def extract_number(image: np.ndarray, orientation: str = "horizontal", ignore: float = None) -> float | None:
#     if image is None or image.size == 0:
#         return None

#     best_text = ""
#     largest_value = -float('inf')

#     if orientation == "vertical":
#         rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
#     elif orientation == "diagonal":
#         rotations = [45, -45]
#     else:
#         rotations = [None]

#     for rotation_code in rotations:
#         rotated_image = image
#         if rotation_code is not None:
#             if orientation == "diagonal":
#                 (h, w) = image.shape[:2]
#                 center = (w // 2, h // 2)
#                 M = cv2.getRotationMatrix2D(center, rotation_code, 1.0)
#                 rotated_image = cv2.warpAffine(image, M, (w, h))
#             else:
#                 rotated_image = cv2.rotate(image, rotation_code)

#         result = OCR_READER.readtext(
#             rotated_image,
#             allowlist='0123456789.,',
#             detail=1,
#             paragraph=False
#         )

#         if result:
#             current_text = "".join([res[1] for res in result])
#             current_text = current_text.replace(',', '.')
#             try:
#                 current_value = float(current_text)
#                 if ignore and current_value == ignore:
#                     continue
                
#                 if current_value > largest_value:
#                     largest_value = current_value
#                     best_text = current_text
#             except (ValueError, TypeError):
#                 continue

#     if best_text:
#         try:
#             return float(best_text)
#         except (ValueError, TypeError):
#             pass

#     return None


# def rotate_angle_image(image: np.ndarray) -> np.ndarray:
#     if image is None or image.size == 0:
#         return image

#     height, width = image.shape[:2]
#     if height > width:
#         return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#     return image


# # def text(pdf_path: str, dpi: int = 300, page_number: int = 0):
# #     """
# #     Load a PDF page, convert it to an image, run OCR to extract numeric values,
# #     overlay detected numbers on the original image, and display the result.

# #     Args:
# #         pdf_path: Absolute or relative path to the input PDF file.
# #         dpi: Rendering DPI for PDF-to-image conversion.
# #         page_number: Zero-based page index to process.
# #     """
# #     # Convert PDF page to image (RGB) and grayscale
# #     img_rgb, _ = convert_to_image(file_path=pdf_path, dpi=dpi, page_number=page_number)

# #     # Run OCR to detect text regions
# #     detections = OCR_READER.readtext(
# #         img_rgb,
# #         allowlist='0123456789.,abcdefghijklmnopqrstuvwxyz',
# #         detail=1,
# #         paragraph=False
# #     )

# #     # Prepare a copy for visualization (OpenCV expects BGR for drawing)
# #     viz_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# #     found_any = False
# #     for det in detections or []:
# #         # det: (box, text, confidence)
# #         box, raw_text, conf = det
# #         # Compute axis-aligned bounding rectangle from quadrilateral box
# #         pts = np.array(box, dtype=np.int32)
# #         x, y, w, h = cv2.boundingRect(pts)
# #         # Crop safely within bounds
# #         H, W = viz_bgr.shape[:2]
# #         x0, y0 = max(0, x), max(0, y)
# #         x1, y1 = min(W, x + w), min(H, y + h)
# #         crop_bgr = viz_bgr[y0:y1, x0:x1]
# #         crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

# #         # Try to parse number from the cropped region
# #         value = extract_number(crop_rgb)
# #         if value is None:
# #             continue

# #         found_any = True
# #         # Draw rectangle and value
# #         cv2.rectangle(viz_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
# #         label = f"{value}"
# #         cv2.putText(viz_bgr, label, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

# #     # Convert back to RGB for matplotlib and show
# #     viz_rgb = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)
# #     plt.figure(figsize=(12, 12))
# #     plt.imshow(viz_rgb)
# #     plt.axis('off')
# #     plt.show()

# #     return found_any


# def _map_point_from_rotated_to_original(pt: tuple[float, float], angle: int, orig_w: int, orig_h: int) -> tuple[float, float]:
#     """
#     Map a point (x, y) from a rotated image back to original image coordinates.
#     angle must be one of {0, 90, 180, 270} (degrees clockwise).
#     Coordinates are (x=col, y=row).
#     """
#     x_r, y_r = float(pt[0]), float(pt[1])
#     if angle == 0:
#         return (x_r, y_r)
#     elif angle == 90:  # rotated = cv2.ROTATE_90_CLOCKWISE
#         # inverse: x_o = y_r, y_o = orig_h - 1 - x_r
#         return (y_r, orig_h - 1 - x_r)
#     elif angle == 180:  # cv2.ROTATE_180
#         # inverse: x_o = orig_w - 1 - x_r, y_o = orig_h - 1 - y_r
#         return (orig_w - 1 - x_r, orig_h - 1 - y_r)
#     elif angle == 270:  # cv2.ROTATE_90_COUNTERCLOCKWISE
#         # inverse: x_o = orig_w - 1 - y_r, y_o = x_r
#         return (orig_w - 1 - y_r, x_r)
#     else:
#         raise ValueError("angle must be in {0,90,180,270}")


# def text(pdf_path: str, dpi: int = 300, page_number: int = 0):
#     """
#     Convert a PDF page to an image and run OCR on four rotations (0,90,180,270).
#     Returns a list of detections (liberal: keeps detected text even if not parsed as number).

#     Each detection is a dict:
#       {
#         'box': [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]  # quadrilateral in ORIGINAL image coords (floats)
#         'bbox': (xmin, ymin, xmax, ymax)              # axis-aligned integer bbox (clamped)
#         'raw_text': str,
#         'conf': float,
#         'angle': int,                                 # rotation (0/90/180/270) at which OCR saw it
#         'value': float | None                         # numeric value via extract_number if found
#       }
#     """
#     # Load image (RGB) from PDF
#     img_rgb, _ = convert_to_image(file_path=pdf_path, dpi=dpi, page_number=page_number)
#     if img_rgb is None or img_rgb.size == 0:
#         return []

#     orig_h, orig_w = img_rgb.shape[:2]

#     # Prepare rotated images
#     rotations = {
#         0: img_rgb,
#         90: cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE),
#         180: cv2.rotate(img_rgb, cv2.ROTATE_180),
#         270: cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE),
#     }

#     colocated_results = []  # store detections mapped to original image coordinates

#     # Run OCR on each rotation and map boxes back
#     for angle, rot_img in rotations.items():
#         try:
#             results = OCR_READER.readtext(
#                 rot_img,
#                 allowlist='0123456789.,abcdefghijklmnopqrstuvwxyz',
#                 detail=1,
#                 paragraph=False
#             )
#         except Exception:
#             results = []

#         for box, raw_text, conf in results or []:
#             # box is a list of 4 points (x, y) in rotated image coords
#             pts_rot = np.array(box, dtype=np.float32)  # shape (4,2)

#             # Map each point back to original image coordinates
#             pts_orig = np.array([
#                 _map_point_from_rotated_to_original((float(x), float(y)), angle, orig_w, orig_h)
#                 for (x, y) in pts_rot
#             ], dtype=np.float32)

#             # Compute integer, clamped axis-aligned bounding box in original coordinates
#             xmin = int(np.floor(np.min(pts_orig[:, 0])))
#             xmax = int(np.ceil(np.max(pts_orig[:, 0])))
#             ymin = int(np.floor(np.min(pts_orig[:, 1])))
#             ymax = int(np.ceil(np.max(pts_orig[:, 1])))

#             # Clamp to image bounds
#             x0 = max(0, xmin)
#             y0 = max(0, ymin)
#             x1 = min(orig_w, xmax)
#             y1 = min(orig_h, ymax)

#             # Validate bbox (guard against inverted / zero-area boxes)
#             if x1 <= x0 or y1 <= y0:
#                 # skip invalid crop (this avoids cvtColor on empty arrays)
#                 continue

#             # Crop from a BGR viz copy for drawing and from an RGB crop used by extract_number
#             viz_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
#             crop_bgr = viz_bgr[y0:y1, x0:x1]
#             if crop_bgr is None or crop_bgr.size == 0:
#                 continue
#             crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

#             # Decide orientation to pass to extract_number (vertical for 90/270)
#             orientation = "vertical" if angle in (90, 270) else "horizontal"
#             value = extract_number(crop_rgb, orientation=orientation)

#             colocated_results.append({
#                 "box": pts_orig.tolist(),
#                 "bbox": (x0, y0, x1, y1),
#                 "raw_text": raw_text,
#                 "conf": float(conf) if conf is not None else None,
#                 "angle": angle,
#                 "value": value
#             })

#     # Visualization: draw all detections on the ORIGINAL image
#     viz_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
#     color_map = {
#         0: (0, 255, 0),    # green
#         90: (0, 0, 255),   # red
#         180: (255, 0, 0),  # blue
#         270: (0, 255, 255) # yellow
#     }

#     for det in colocated_results:
#         x0, y0, x1, y1 = det["bbox"]
#         color = color_map.get(det["angle"], (0, 255, 0))
#         cv2.rectangle(viz_bgr, (x0, y0), (x1, y1), color, 2)
#         label = det["raw_text"]
#         if det["value"] is not None:
#             # show number if we parsed one
#             label = f"{det['value']}"
#         text_y = max(12, y0 - 5)
#         cv2.putText(viz_bgr, label, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # Show merged result
#     viz_rgb = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(12, 12))
#     plt.imshow(viz_rgb)
#     plt.axis('off')
#     plt.show()

#     # Return unified list of detections (liberal representation)
#     return colocated_results








import math
import re
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from src.geometry.objects import TextType, TextStructure, TextData


from itertools import product
from tqdm import tqdm


import argparse
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from typing import List, Tuple
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

ocr = PaddleOCR(use_textline_orientation=True, lang="pl", use_doc_unwarping=False, use_doc_orientation_classify=False)


# def rotate_box_back(box, angle, tile_w, tile_h, orientation):
#     """Rotate OCR box back from rotated tile into original coordinates."""
#     pts = np.array(box, dtype=np.float32)
#     if angle == 90:
#         # Swap x,y and mirror
#         pts = np.stack([pts[:, 1], tile_w - pts[:, 0]], axis=1)
#     elif angle == 180:
#         pts = np.stack([tile_w - pts[:, 0], tile_h - pts[:, 1]], axis=1)
#     elif angle == 270:
#         pts = np.stack([tile_h - pts[:, 1], pts[:, 0]], axis=1)
#     return pts.tolist()



def rotate_box_back(
    box: List[List[float]], 
    angle: int, 
    tile_w: int, 
    tile_h: int, 
    orientation: str | int
) -> List[List[float]]:
    """
    Rotates an OCR bounding box from a rotated tile back to original image coordinates,
    ensuring the points start at the semantic top-left of the text.

    Args:
        box: The bounding box coordinates from the OCR result.
        angle: The clockwise rotation angle applied to the tile (0, 90, 180, 270).
        tile_w: The width of the tile before rotation.
        tile_h: The height of the tile before rotation.
        orientation: The orientation of the detected text. "1" signifies that the
                     text is upside down. "0" or any other value assumes the text is upright.

    Returns:
        A list of 4 [x, y] points representing the bounding box in the original
        coordinate system, ordered clockwise starting from the text's top-left corner.
    """
    # 1. Perform the geometric rotation of coordinates
    box = list(box)
    sorted_box = sorted(box, key=lambda x: x[1])[:2]
    sorted_box = sorted(sorted_box, key=lambda x: x[0])
    min_index = sorted_box.index(sorted_box[0])
    box = box[min_index:] + box[:min_index]

    if np.linalg.norm(np.array(box[0])-np.array(box[1])) < np.linalg.norm(np.array(box[0])-np.array(box[-1])):
        box = box[1:] + box[:1]
    
    pts = np.array(box, dtype=np.float32)
    if angle == 90:
        # Inverse transform for 90-degree clockwise rotation
        pts = np.stack([pts[:, 1], tile_w - pts[:, 0]], axis=1)
    elif angle == 180:
        # Inverse transform for 180-degree rotation
        pts = np.stack([tile_w - pts[:, 0], tile_h - pts[:, 1]], axis=1)
    elif angle == 270:
        # Inverse transform for 270-degree clockwise rotation
        pts = np.stack([tile_h - pts[:, 1], pts[:, 0]], axis=1)
    
    final_points = pts.tolist()

    # 2. Re-order points based on text orientation
    try:
        # Ensure orientation can be compared as an integer
        orientation = int(orientation)
    except (ValueError, TypeError):
        orientation = 0
        
    # If orientation is "1", the text is upside down. The OCR's top-left (point 0)
    # is the text's semantic bottom-right. The text's actual top-left is the OCR's
    # bottom-right (point 2). We must shift the list to start with point 2.
    if orientation == 1 and len(final_points) == 4:
        # The original clockwise order is [P0, P1, P2, P3].
        # The new required clockwise order is [P2, P3, P0, P1].
        return final_points[2:] + final_points[:2]

    # For orientation "0" (or any other case), the order is assumed to be correct.
    return final_points





# helper to unify overlapping boxes
def unify_boxes(results, iou_threshold=0.3):
    """Merge duplicates using polygon IoU and keep best unique boxes."""
    merged = {}
    for i, (box, text, score) in enumerate(results):
        if not text.strip(): 
            continue
        poly = Polygon(box)
        duplicate = False
        for j, (obox, otext, oscore) in merged.items():
            opoly = Polygon(obox)
            if poly.intersects(opoly):
                inter = poly.intersection(opoly).area
                union = poly.union(opoly).area
                iou = inter / union
                if iou > iou_threshold: # and text.lowet().strip() == otext:
                    duplicate = True
                    break
        if not duplicate:
            merged[i] = (box, text.lower().strip(), score)
        else:
            if score  > oscore:
                merged[j] = (box, text.lower().strip(), score)
    return merged.values()

def process_image_in_tiles(image, ocr_engine=ocr, tile_size=1024, overlap=100):

    h, w, _ = image.shape
    all_results = []

    step_size = tile_size - overlap

    total = (h // step_size + 1) * (w // step_size + 1)
    
    originals = {}

    for y, x in tqdm(product(list(range(0, h, step_size)), list(range(0, w, step_size))), total=total):
        y_end = min(y + tile_size, h)
        x_end = min(x + tile_size, w)
        tile = image[y:y_end, x:x_end]

        for angle in [0, 90, 180, 270]:
            # Rotate tile
            if angle == 0:
                rotated = tile
            elif angle == 90:
                rotated = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(tile, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)

            tile_h, tile_w = rotated.shape[:2]

            tile_results = ocr_engine.predict(rotated)
            if tile_results and tile_results[0] is not None:
                originals[(x, y, angle)] = tile_results[0]
                for text, box, orientation, score in zip(tile_results[0]["rec_texts"], tile_results[0]["rec_polys"], 
                tile_results[0].get("textline_orientation_angles", []), tile_results[0].get("rec_scores", [])):
                    # rotate box back to original tile orientation
                    corrected_box = rotate_box_back(box, angle, tile_w, tile_h, orientation)

                    # shift to global coords
                    adjusted_box = []
                    for px, py in corrected_box:
                        adj_x = float(px) + x
                        adj_y = float(py) + y
                        adjusted_box.append([adj_x, adj_y])

                    all_results.append((adjusted_box, text, score))
                r = tile_results[0]
                # r.save_to_img("output")

    # unify overlapping/duplicate results
    final_results = unify_boxes(all_results)
    return final_results_to_text_data(final_results), originals


# --- Helper functions for the conversion ---

def _calculate_orientation(box: List[Tuple[float, float]]) -> float:
    """Calculates the angle of the longer side of the bounding box."""
    if len(box) < 4:
        return 0.0

    # Get the first three points of the bounding box
    p0, p1, p2 = box[0], box[1], box[2]

    # Calculate the squared length of the two adjacent sides
    side1_sq_len = (p1[0] - p0[0])**2 + (p1[1] - p0[1])**2
    side2_sq_len = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2

    # Determine the vector of the longer side
    if side1_sq_len >= side2_sq_len:
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
    else:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
    
    # Calculate the angle in degrees
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def _classify_text_type(text: str) -> TextType:
    """Classifies the text content into predefined categories."""
    cleaned_text = text.strip()

    if not cleaned_text:
        return TextType.NATURAL_TEXT

    if '%' in cleaned_text:
        return TextType.PERCENTAGE
    
    if '°' in cleaned_text:
        return TextType.ANGLE

    # Regex to match integers or decimals (with either '.' or ',')
    # Allows for optional leading sign
    numerical_pattern = re.compile(r'^-?\d+([.,]\d+)?$')
    if numerical_pattern.match(cleaned_text):
        return TextType.NUMERICAL
    
    return TextType.NATURAL_TEXT


# --- Main Conversion Function ---

def final_results_to_text_data(final_results: List[Tuple[List[List[float]], str]]) -> List[TextData]:
    """
    Converts the raw OCR output into a structured list of TextData objects.

    Args:
        final_results: A list of tuples, where each tuple contains:
                       - A bounding box (list of 4 [x, y] points).
                       - The recognized text string.

    Returns:
        A list of Pydantic TextData objects.
    """
    text_data_list = []
    for box, text_content, score in final_results:
        # Ensure the box format is a list of tuples
        bounding_box_tuples = [tuple(point) for point in box]

        # Calculate orientation from the box geometry
        orientation = _calculate_orientation(bounding_box_tuples)

        # Classify text content
        text_type = _classify_text_type(text_content)

        # Create the TextData object
        # NOTE: We assume single-line structure as OCR engines typically
        # return text on a line-by-line basis.
        text_data_item = TextData(
            bounding_box=bounding_box_tuples,
            text=text_content.strip(),
            orientation=orientation,
            text_type=text_type,
            structure=TextStructure.SINGLE_LINE,
        )
        text_data_list.append(text_data_item)

    return text_data_list