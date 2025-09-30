# import math
# from collections import defaultdict
#
# import cv2
# import numpy as np
#
#
#
# def get_robust_thickness(line, binary_image):
#     p1, p2 = line['endpoints'];
#     angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
#     perp_angle_rad = angle_rad + math.pi / 2
#     mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
#     thickness = 0;
#     max_check = 15
#     for i in range(max_check):
#         x, y = int(mid_x + i * math.cos(perp_angle_rad)), int(mid_y + i * math.sin(perp_angle_rad))
#         if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1] and binary_image[y, x] != 0:
#             thickness += 1
#         else:
#             break
#     for i in range(1, max_check):
#         x, y = int(mid_x - i * math.cos(perp_angle_rad)), int(mid_y - i * math.sin(perp_angle_rad))
#         if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1] and binary_image[y, x] != 0:
#             thickness += 1
#         else:
#             break
#     return max(1, thickness)
#
#
# def classify_thickness(width):
#     if width <= THIN_NORMAL_THRESHOLD:
#         return "thin"
#     elif width <= NORMAL_THICK_THRESHOLD:
#         return "normal"
#     else:
#         return "thick"
#
# def point_distance(p1, p2):
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
#
#
#
# def classify_lines(lines, img_original, img_gray, img_denoised):
#     mask_line = np.zeros(img_denoised.shape, dtype=np.uint8)
#     cv2.line(
#         mask_line,
#         line_obj["endpoints"][0],
#         line_obj["endpoints"][1],
#         255,
#         3,
#     )
#     mean_hsv = cv2.mean(img_hsv, mask=mask_line)[:3]
#     mean_hsv = np.array(mean_hsv, dtype=np.uint8)
#
#     if (
#             YELLOW_LOWER[0] <= mean_hsv[0] <= UPPER_YELLOW[0]
#             and YELLOW_LOWER[1] <= mean_hsv[1] <= UPPER_YELLOW[1]
#             and YELLOW_LOWER[2] <= mean_hsv[2] <= UPPER_YELLOW[2]
#     ):
#         line_obj["type"] = "yellow"
#         final_lines_to_draw.append(
#             {"points": (p[0], p[1], p[2], p[3]), "type": "yellow"}
#         )
#         continue
#
#     # Otherwise, thickness-based classification
#     thickness = get_robust_thickness(line_obj, img_denoised)
#     line_obj["type"] = classify_thickness(thickness)
#     lines_by_type[line_obj["type"]].append(line_obj)
#
#
# def detect_lines(img_original, img_gray, img_denoised, min_line_len_px=None):
#     final_lines_to_draw = []
#
#     # 1. Detect all lines on the binary image
#     lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
#     lines_lsd, _, _, _ = lsd.detect(img_denoised)
#
#     lines_by_type = defaultdict(list)
#
#     if lines_lsd is not None:
#         # Prepare HSV for color classification
#         img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
#
#         for seg in lines_lsd:
#             p = seg[0]
#             length = math.hypot(p[2] - p[0], p[3] - p[1])
#             if min_line_len_px and length < min_line_len_px:
#                 continue
#
#             line_obj = {
#                 "endpoints": ((int(p[0]), int(p[1])), (int(p[2]), int(p[3]))),
#                 "angle": math.degrees(math.atan2(p[3] - p[1], p[2] - p[0])),
#                 "length": length,
#             }
#


import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from src.geometry.objects import Interval

# ==============================================================================
# 1. Configuration Constants
# ==============================================================================

# Thickness thresholds in pixels for classifying gray/black lines
THIN_NORMAL_THRESHOLD = 2
NORMAL_THICK_THRESHOLD = 5

# Color range for 'yellow' in HSV color space
# Hue values are typically in the range [0, 179] for OpenCV
YELLOW_LOWER_HSV = np.array([20, 100, 100])
YELLOW_UPPER_HSV = np.array([30, 255, 255])

COLOR_MAP = {
    "thin": (0, 255, 0),
    "normal": (255, 0, 0),
    "thick": (0, 0, 255),
    "yellow": (0, 255, 255),
    "dashed": (128, 0, 128),
}
# ==============================================================================
# 2. Pydantic Data Structure
# ==============================================================================


# ==============================================================================
# 3. Core Functions (As per requirements)
# ==============================================================================


def detect_lines(
    binary_image: np.ndarray, min_line_len_px: Optional[float] = 40.0
) -> List[Interval]:
    """
    Detects all line segments in a binary image using LSD (Line Segment Detector).

    Args:
        binary_image: The input single-channel binary image where lines are non-zero.
        min_line_len_px: The minimum length in pixels for a line to be considered.

    Returns:
        A list of Interval objects, each representing a detected line.
    """
    detected_intervals = []

    # Initialize the Line Segment Detector
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines_lsd, _, _, _ = lsd.detect(binary_image)

    if lines_lsd is not None:
        for line_segment in lines_lsd:
            p = line_segment[0]
            p1 = (int(p[0]), int(p[1]))
            p2 = (int(p[2]), int(p[3]))

            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

            # Filter out lines that are too short
            if min_line_len_px and length < min_line_len_px:
                continue

            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

            # Create an Interval object with the mandatory fields
            interval = Interval(
                endpoints=(p1, p2),
                angle=angle,
                length=length,
            )
            detected_intervals.append(interval)

    return detected_intervals


def classify_lines(
    intervals: List[Interval], original_rgb_image: np.ndarray, binary_image: np.ndarray
) -> List[Interval]:
    """
    Classifies a list of intervals by color and thickness, updating them in place.

    Args:
        intervals: A list of Interval objects to classify.
        original_bgr_image: The original 3-channel BGR color image.
        binary_image: The binary image used for accurate thickness measurement.

    Returns:
        The list of Interval objects with updated 'classification' and 'thickness' fields.
    """
    img_hsv = cv2.cvtColor(original_rgb_image, cv2.COLOR_RGB2HSV)

    for interval in intervals:
        # --- 1. Color Classification ---
        # Create a mask for the current line to sample its color
        mask = np.zeros(binary_image.shape, dtype=np.uint8)
        # Use a slightly thicker line for the mask to grab more color information
        cv2.line(mask, interval.endpoints[0], interval.endpoints[1], 255, thickness=5)

        # Calculate the mean color within the masked area
        mean_hsv = cv2.mean(img_hsv, mask=mask)

        # Check if the mean color falls within the yellow range
        is_yellow = cv2.inRange(
            np.array([[mean_hsv[:3]]], dtype=np.uint8),
            YELLOW_LOWER_HSV,
            YELLOW_UPPER_HSV,
        )

        if is_yellow[0][0] == 255:
            interval.classification = "yellow"
            continue  # Move to the next line if color is identified

        # --- 2. Thickness Classification ---
        # If not classified by color, classify it by thickness
        thickness = get_robust_thickness(interval.endpoints, binary_image)
        interval.thickness = thickness
        interval.classification = classify_thickness_category(
            thickness, scale=max(binary_image.shape)
        )

    return intervals


# ==============================================================================
# 4. Helper Functions
# ==============================================================================


def get_robust_thickness(
    endpoints: Tuple[Tuple[int, int], Tuple[int, int]], binary_image: np.ndarray
) -> int:
    """
    Calculates line thickness by checking pixels perpendicular to its midpoint.

    Args:
        endpoints: A tuple containing the (x1, y1) and (x2, y2) points of the line.
        binary_image: The binary image on which the line exists.

    Returns:
        The calculated thickness in pixels.
    """
    p1, p2 = endpoints
    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    perp_angle_rad = angle_rad + math.pi / 2

    mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

    thickness = 0
    max_check_dist = 20  # Check up to 20 pixels on each side

    # Check in the positive perpendicular direction from the midpoint
    for i in range(max_check_dist):
        x = int(round(mid_x + i * math.cos(perp_angle_rad)))
        y = int(round(mid_y + i * math.sin(perp_angle_rad)))
        if (
            0 <= y < binary_image.shape[0]
            and 0 <= x < binary_image.shape[1]
            and binary_image[y, x] != 0
        ):
            thickness += 1
        else:
            break

    # Check in the negative perpendicular direction
    for i in range(1, max_check_dist):
        x = int(round(mid_x - i * math.cos(perp_angle_rad)))
        y = int(round(mid_y - i * math.sin(perp_angle_rad)))
        if (
            0 <= y < binary_image.shape[0]
            and 0 <= x < binary_image.shape[1]
            and binary_image[y, x] != 0
        ):
            thickness += 1
        else:
            break

    return max(1, thickness)


def classify_thickness_category(width: int, scale: int | None = None) -> str:
    """Classifies a pixel width into a predefined category."""
    thin_threshold = THIN_NORMAL_THRESHOLD
    norm_threshold = NORMAL_THICK_THRESHOLD
    if scale:
        thin_threshold = 0.0003 * scale
        norm_threshold = 0.0006 * scale
    if width <= thin_threshold:
        return "thin"
    elif width <= norm_threshold:
        return "normal"
    else:
        return "thick"


# ==============================================================================
# 5. Example Usage
# ==============================================================================
if __name__ == "__main__":
    # --- Setup: Create a dummy image for demonstration ---
    img_bgr = np.full((300, 500, 3), (255, 255, 255), dtype=np.uint8)

    # Draw lines to be detected: (image, p1, p2, BGR_color, thickness)
    cv2.line(img_bgr, (50, 50), (450, 50), (0, 0, 0), 2)  # thin
    cv2.line(img_bgr, (50, 100), (450, 100), (100, 100, 100), 4)  # normal
    cv2.line(img_bgr, (50, 150), (450, 150), (0, 0, 0), 8)  # thick
    cv2.line(img_bgr, (50, 200), (450, 200), (0, 255, 255), 5)  # yellow
    cv2.line(img_bgr, (50, 220), (250, 280), (128, 128, 128), 1)  # thin, diagonal

    # --- Pre-processing: Create a binary image for detection ---
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)

    # --- Step 1: Detect all line intervals ---
    print("1. Detecting lines...")
    all_intervals = detect_lines(img_binary, min_line_len_px=40)
    print(f"   ✅ Found {len(all_intervals)} line(s).")
    for i, interval in enumerate(all_intervals):
        print(
            f"   - Initial Line {i}: Length={interval.length:.1f}px, Angle={interval.angle:.1f}°"
        )

    # --- Step 2: Classify the detected intervals ---
    print("\n2. Classifying lines by color and thickness...")
    classified_intervals = classify_lines(all_intervals, img_bgr, img_binary)
    print("   ✅ Classification complete.")

    # --- Step 3: Display Results ---
    print("\n3. Final Classification Results:")
    for i, interval in enumerate(classified_intervals):
        print(
            f"   - Line {i}: Classification='{interval.classification}', "
            f"Thickness={interval.thickness}px, "
            f"Endpoints={interval.endpoints}"
        )

    # --- Visualization ---
    img_viz = img_bgr.copy()
    for interval in classified_intervals:
        p1, p2 = interval.endpoints
        # Draw a green line over the detected segment
        cv2.line(img_viz, p1, p2, (0, 255, 0), 2)
        # Add a text label with the classification info
        label = f"{interval.classification} (T:{interval.thickness})"
        text_pos = (p1[0] + 5, p1[1] - 8)
        cv2.putText(
            img_viz, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

    # To display the images, uncomment the following lines:
    # cv2.imshow("Original Image with Lines", img_bgr)
    # cv2.imshow("Binary Image for Detection", img_binary)
    # cv2.imshow("Detection & Classification Results", img_viz)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
