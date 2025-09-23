# file_path = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow/6.pdf"
file_path = "ex1.pdf"

# import fitz  # PyMuPDF
# import cv2
# import numpy as np
# import math
#
# # ---------- Configuration & Constants ----------
#
# PDF_RENDER_DPI = 300
#
# # Thickness thresholds for classifying lines. Tuned for 300 DPI.
# # Thick lines (walls) are typically > 4.0 pixels at this resolution.
# THICK_THRESHOLD = 4.0
#
# # All other lines will be considered "thin" for simplicity and clarity.
# # Increased minimum length to effectively filter out text characters.
# MIN_LINE_LEN_PX = 25
#
# # HSV color range for yellow highlights
# YELLOW_LOWER = np.array([15, 50, 150])
# YELLOW_UPPER = np.array([45, 255, 255])
#
# # Updated color map for clear visualization
# COLOR_MAP = {
#     "thin": (0, 255, 0),  # Green
#     "thick": (0, 0, 255),  # Red
#     "yellow": (0, 255, 255),  # Yellow
# }
#
#
# # ---------- Utility Functions ----------
#
# def convert_to_image(file_path, dpi=300):
#     """Handles PDF/image loading, returning grayscale and color images at high DPI."""
#     try:
#         if file_path.lower().endswith('.pdf'):
#             doc = fitz.open(file_path)
#             page = doc[0]
#             zoom = dpi / 72
#             matrix = fitz.Matrix(zoom, zoom)
#             pix = page.get_pixmap(matrix=matrix, alpha=False)
#             img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             return gray, img
#         img = cv2.imread(file_path)
#         if img is None: raise FileNotFoundError(f"File not found: {file_path}")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return gray, img
#     except Exception as e:
#         print(f"Error converting file '{file_path}': {e}")
#         return None, None
#
#
# def filter_lines_by_length(lines, min_length):
#     """Filters a list of detected lines, removing any shorter than a minimum threshold."""
#     if lines is None:
#         return []
#     filtered_lines = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         length = math.hypot(x2 - x1, y2 - y1)
#         if length > min_length:
#             filtered_lines.append(line)
#     return filtered_lines
#
#
# # ---------- Main Analysis Function ----------
#
# def analyze_plan_from_scratch(file_path, out_path="out_classified_new.jpg"):
#     """
#     Analyzes a building plan by separating it into layers (color, thick, thin)
#     and detecting lines in each, without using OCR.
#     """
#     # 1. Load the image at high resolution.
#     gray_img, original_img = convert_to_image(file_path, dpi=PDF_RENDER_DPI)
#     if gray_img is None:
#         return {"error": "Failed to load the file."}
#
#     all_lines_info = []
#
#     # --- Step A: Detect and Remove Yellow Highlighted Lines ---
#     img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
#     mask_yellow = cv2.inRange(img_hsv, YELLOW_LOWER, YELLOW_UPPER)
#
#     yellow_lines_raw = cv2.HoughLinesP(mask_yellow, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
#     if yellow_lines_raw is not None:
#         for line in yellow_lines_raw:
#             all_lines_info.append({"points": line[0], "type": "yellow"})
#
#     # Create an inverse mask to "erase" yellow areas for subsequent steps.
#     mask_yellow_inv = cv2.bitwise_not(cv2.dilate(mask_yellow, np.ones((15, 15), np.uint8)))
#     gray_img = cv2.bitwise_and(gray_img, gray_img, mask=mask_yellow_inv)
#
#     # --- Step B: Detect Thick Lines (Walls) ---
#     # Binarize the image to isolate foreground elements.
#     thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
#
#     # Use morphological operations to isolate only the thickest lines.
#     thick_kernel = np.ones((3, 3), np.uint8)
#     # Erode away thin lines, leaving only thick ones.
#     eroded_img = cv2.erode(thresh_img, thick_kernel, iterations=2)
#
#     thick_lines_raw = cv2.HoughLinesP(eroded_img, 1, np.pi / 180, threshold=50, minLineLength=MIN_LINE_LEN_PX,
#                                       maxLineGap=10)
#     if thick_lines_raw is not None:
#         for line in thick_lines_raw:
#             all_lines_info.append({"points": line[0], "type": "thick"})
#
#     # "Erase" the detected thick lines from the main thresholded image to avoid re-detection.
#     if thick_lines_raw is not None:
#         for line in thick_lines_raw:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(thresh_img, (x1, y1), (x2, y2), 0, 10)  # Draw black lines over them
#
#     # --- Step C: Detect Thin Lines (Roof Tiles, Details) ---
#     # On the remaining image (with thick lines removed), detect everything else.
#     # The Line Segment Detector is excellent for the numerous thin lines.
#     lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
#     thin_lines_lsd, _, _, _ = lsd.detect(thresh_img)
#
#     if thin_lines_lsd is not None:
#         # Filter out short lines which are likely text fragments
#         for line in thin_lines_lsd:
#             x1, y1, x2, y2 = map(int, line[0])
#             if math.hypot(x2 - x1, y2 - y1) > MIN_LINE_LEN_PX:
#                 all_lines_info.append({"points": (x1, y1, x2, y2), "type": "thin"})
#
#     # --- Step D: Draw Results ---
#     output_img = original_img.copy()
#     print(f"Detected {len(all_lines_info)} total line segments.")
#     for line_info in all_lines_info:
#         x1, y1, x2, y2 = line_info["points"]
#         color = COLOR_MAP.get(line_info["type"])
#         cv2.line(output_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
#
#     cv2.imwrite(out_path, output_img)
#     print(f"Analysis complete. New image saved to {out_path}")
#     return {"count": len(all_lines_info)}

# import fitz  # PyMuPDF
# import cv2
# import numpy as np
# import math
#
# # ---------- Configuration & Constants ----------
#
# PDF_RENDER_DPI = 300
# MIN_LINE_LEN_PX = 20  # Minimum length to consider a line segment valid
#
# # New 4-tier classification thresholds based on pixel width at 300 DPI
# THIN_NORMAL_THRESHOLD = 3  # Lines <= 2.5px are thin
# NORMAL_THICK_THRESHOLD = 4  # Lines > 4.5px are thick
#
# # Line merging parameters
# MERGE_ANGLE_TOLERANCE_DEGREES = 2.0  # Max angle difference to be considered collinear
# MERGE_DISTANCE_TOLERANCE_PIXELS = 5.0  # Max distance between endpoints to merge
#
# # Updated color map including 'normal'
# COLOR_MAP = {
#     "thin": (0, 255, 0),  # Green
#     "normal": (255, 0, 0),  # Blue
#     "thick": (0, 0, 255),  # Red
#     "yellow": (0, 255, 255),  # Yellow (Highest Priority)
# }
#
# # HSV color range for yellow highlights
# YELLOW_LOWER = np.array([15, 50, 150])
# YELLOW_UPPER = np.array([45, 255, 255])
#
#
# # ---------- Utility Functions ----------
#
# def convert_to_image(file_path, dpi=300):
#     """Handles PDF/image loading, returning grayscale and color images at high DPI."""
#     try:
#         if file_path.lower().endswith('.pdf'):
#             doc = fitz.open(file_path)
#             page = doc[0]
#             zoom = dpi / 72
#             matrix = fitz.Matrix(zoom, zoom)
#             pix = page.get_pixmap(matrix=matrix, alpha=False)
#             img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             return gray, img
#         img = cv2.imread(file_path)
#         if img is None: raise FileNotFoundError(f"File not found: {file_path}")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return gray, img
#     except Exception as e:
#         print(f"Error converting file '{file_path}': {e}")
#         return None, None
#
#
# def classify_thickness(width):
#     """Classifies a line's width into 'thin', 'normal', or 'thick'."""
#     if width <= THIN_NORMAL_THRESHOLD:
#         return "thin"
#     elif width <= NORMAL_THICK_THRESHOLD:
#         return "normal"
#     else:
#         return "thick"
#
#
# def get_line_angle(line_points):
#     """Calculates the angle of a line in degrees."""
#     x1, y1, x2, y2 = line_points
#     return math.degrees(math.atan2(y2 - y1, x2 - x1))
#
#
# def point_distance(p1, p2):
#     """Calculates the Euclidean distance between two points."""
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
#
#
# # ---------- Line Merging Logic ----------
#
# def merge_lines(lines_to_merge):
#     """
#     Merges collinear and close line segments of the same classification type.
#     """
#     if not lines_to_merge:
#         return []
#
#     merged_flags = [False] * len(lines_to_merge)
#     merged_lines_final = []
#
#     for i in range(len(lines_to_merge)):
#         if merged_flags[i]:
#             continue
#
#         base_line = lines_to_merge[i]
#         current_type = base_line['type']
#
#         # Start a group of lines to be merged with the base line
#         line_group_points = list(base_line['endpoints'])
#         merged_flags[i] = True
#
#         for j in range(i + 1, len(lines_to_merge)):
#             if merged_flags[j]:
#                 continue
#
#             candidate_line = lines_to_merge[j]
#
#             # Condition 1: Must be the same type (e.g., 'thin' with 'thin')
#             if candidate_line['type'] != current_type:
#                 continue
#
#             # Condition 2: Must be nearly collinear
#             angle_diff = abs(base_line['angle'] - candidate_line['angle'])
#             # Handle angle wrap-around (e.g., 179 deg and -179 deg are similar)
#             if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES:
#                 continue
#
#             # Condition 3: Endpoints must be close
#             p1, p2 = base_line['endpoints']
#             q1, q2 = candidate_line['endpoints']
#             if min(point_distance(p1, q1), point_distance(p1, q2), point_distance(p2, q1),
#                    point_distance(p2, q2)) > MERGE_DISTANCE_TOLERANCE_PIXELS:
#                 continue
#
#             # If all conditions pass, add the candidate's points to the group and mark as merged
#             line_group_points.extend(candidate_line['endpoints'])
#             merged_flags[j] = True
#
#         # For the collected group, find the two outermost points
#         max_dist = -1
#         final_p1, final_p2 = None, None
#         for p_start_idx in range(len(line_group_points)):
#             for p_end_idx in range(p_start_idx + 1, len(line_group_points)):
#                 dist = point_distance(line_group_points[p_start_idx], line_group_points[p_end_idx])
#                 if dist > max_dist:
#                     max_dist = dist
#                     final_p1 = line_group_points[p_start_idx]
#                     final_p2 = line_group_points[p_end_idx]
#
#         if final_p1 and final_p2:
#             merged_lines_final.append({
#                 "points": (final_p1[0], final_p1[1], final_p2[0], final_p2[1]),
#                 "type": current_type
#             })
#
#     return merged_lines_final
#
#
# # ---------- Main Analysis Function ----------
#
# def analyze_plan_advanced(file_path, out_path="out_final.jpg"):
#     """
#     Main pipeline to detect, classify, merge, and draw lines.
#     """
#     # 1. Load image
#     gray_img, original_img = convert_to_image(file_path, dpi=PDF_RENDER_DPI)
#     if gray_img is None: return {"error": "Failed to load."}
#
#     final_lines_to_draw = []
#
#     # 2. Process colorful lines first (highest priority)
#     img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
#     mask_yellow = cv2.inRange(img_hsv, YELLOW_LOWER, YELLOW_UPPER)
#     yellow_lines_raw = cv2.HoughLinesP(mask_yellow, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
#     if yellow_lines_raw is not None:
#         for line in yellow_lines_raw:
#             final_lines_to_draw.append({"points": line[0], "type": "yellow"})
#
#     # 3. Mask out colored areas for grayscale processing
#     mask_yellow_inv = cv2.bitwise_not(cv2.dilate(mask_yellow, np.ones((15, 15), np.uint8)))
#     gray_masked = cv2.bitwise_and(gray_img, gray_img, mask=mask_yellow_inv)
#
#     # 4. Detect all grayscale lines using LSD to get thickness info
#     lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
#     lines_lsd, widths, _, _ = lsd.detect(gray_masked)
#
#     grayscale_lines_to_process = []
#     if lines_lsd is not None:
#         for i, line_segment in enumerate(lines_lsd):
#             p = line_segment[0]
#             if math.hypot(p[2] - p[0], p[3] - p[1]) < MIN_LINE_LEN_PX:
#                 continue
#
#             line_type = classify_thickness(widths[i][0])
#             grayscale_lines_to_process.append({
#                 'endpoints': ((int(p[0]), int(p[1])), (int(p[2]), int(p[3]))),
#                 'type': line_type,
#                 'angle': get_line_angle(p)
#             })
#
#     # 5. Merge the detected grayscale lines
#     merged_grayscale_lines = merge_lines(grayscale_lines_to_process)
#     final_lines_to_draw.extend(merged_grayscale_lines)
#
#     # 6. Draw the final, merged lines on the image
#     output_img = original_img.copy()
#     for line_info in final_lines_to_draw:
#         x1, y1, x2, y2 = map(int, line_info["points"])
#         color = COLOR_MAP.get(line_info["type"])
#         cv2.line(output_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
#
#     cv2.imwrite(out_path, output_img)
#     print(f"Analysis complete. Drew {len(final_lines_to_draw)} final lines to {out_path}")
#     return {"count": len(final_lines_to_draw)}
#
#
# # ---------- Execution ----------
# if __name__ == "__main__":
#     file_path = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow/6.pdf"
#     analyze_plan_advanced(file_path)
# #
# # # ---------- Execution ----------
# # if __name__ == "__main__":
# #     file_path = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow/6.pdf"
# #     analyze_plan_from_scratch(file_path)

# import fitz  # PyMuPDF
# import cv2
# import numpy as np
# import math
# from collections import defaultdict
#
# # ---------- Configuration & Constants ----------
#
# PDF_RENDER_DPI = 300
# MIN_LINE_LEN_PX = 15  # Allow shorter segments to be candidates for dashed lines
#
# # Classification thresholds
# THIN_NORMAL_THRESHOLD = 2.8
# NORMAL_THICK_THRESHOLD = 4.5
#
# # Merging and Grouping parameters
# MERGE_DISTANCE_TOLERANCE_PIXELS = 5.0  # Updated as per your feedback
# MERGE_ANGLE_TOLERANCE_DEGREES = 2.0
#
# DASHED_GROUP_MIN_LINES = 3  # A pattern needs at least 3 dashes
# DASHED_LENGTH_TOLERANCE = 0.5  # Dashes can be +/- 50% of the seed dash length
# DASHED_GAP_TOLERANCE = 0.5  # Gaps can be +/- 50% of the seed gap length
# DASHED_MAX_GAP_FACTOR = 3  # A gap can't be more than 3x a dash length
#
# # Color map with purple for dashed lines
# COLOR_MAP = {
#     "thin": (0, 255, 0),
#     "normal": (255, 0, 0),
#     "thick": (0, 0, 255),
#     "yellow": (0, 255, 255),
#     "dashed": (128, 0, 128),  # Purple for dashed lines
# }
#
# # HSV color range for yellow highlights
# YELLOW_LOWER = np.array([15, 50, 150])
# UPPER_YELLOW = np.array([45, 255, 255])
#
#
# # ---------- Utility Functions ----------
#
# def convert_to_image(file_path, dpi=300):
#     # This function remains the same as before
#     try:
#         if file_path.lower().endswith('.pdf'):
#             doc = fitz.open(file_path)
#             page = doc[0]
#             zoom = dpi / 72;
#             matrix = fitz.Matrix(zoom, zoom)
#             pix = page.get_pixmap(matrix=matrix, alpha=False)
#             img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             return gray, img, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#                                                     21, 10)
#         img = cv2.imread(file_path)
#         if img is None: raise FileNotFoundError(f"File not found: {file_path}")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return gray, img, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21,
#                                                 10)
#     except Exception as e:
#         print(f"Error converting file '{file_path}': {e}")
#         return None, None, None
#
#
# def get_robust_thickness(line, binary_image):
#     """Measures thickness by checking perpendicular pixels on a binary image."""
#     p1, p2 = line['endpoints']
#     angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
#     perp_angle_rad = angle_rad + math.pi / 2
#
#     mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
#
#     thickness = 0
#     max_check = 15  # Check up to 15 pixels away
#
#     # Check one side
#     for i in range(max_check):
#         x = int(mid_x + i * math.cos(perp_angle_rad))
#         y = int(mid_y + i * math.sin(perp_angle_rad))
#         if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1] and binary_image[y, x] != 0:
#             thickness += 1
#         else:
#             break
#     # Check other side
#     for i in range(1, max_check):
#         x = int(mid_x - i * math.cos(perp_angle_rad))
#         y = int(mid_y - i * math.sin(perp_angle_rad))
#         if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1] and binary_image[y, x] != 0:
#             thickness += 1
#         else:
#             break
#
#     return max(1, thickness)  # Return at least 1
#
#
# def classify_thickness(width):
#     # This function remains the same
#     if width <= THIN_NORMAL_THRESHOLD:
#         return "thin"
#     elif width <= NORMAL_THICK_THRESHOLD:
#         return "normal"
#     else:
#         return "thick"
#
#
# def point_distance(p1, p2):
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
#
#
# # ---------- Core Logic Functions ----------
#
# def merge_lines(lines_to_merge):
#     # This function is largely the same but uses the updated MERGE_DISTANCE_TOLERANCE_PIXELS
#     if not lines_to_merge: return []
#     merged_flags = [False] * len(lines_to_merge)
#     merged_lines_final = []
#     for i in range(len(lines_to_merge)):
#         if merged_flags[i]: continue
#         base_line = lines_to_merge[i];
#         current_type = base_line['type']
#         line_group_points = list(base_line['endpoints'])
#         merged_flags[i] = True
#         for j in range(i + 1, len(lines_to_merge)):
#             if merged_flags[j]: continue
#             candidate_line = lines_to_merge[j]
#             if candidate_line['type'] != current_type: continue
#             angle_diff = abs(base_line['angle'] - candidate_line['angle'])
#             if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES: continue
#             p1, p2 = base_line['endpoints'];
#             q1, q2 = candidate_line['endpoints']
#             if min(point_distance(p1, q1), point_distance(p1, q2), point_distance(p2, q1),
#                    point_distance(p2, q2)) > MERGE_DISTANCE_TOLERANCE_PIXELS: continue
#             line_group_points.extend(candidate_line['endpoints']);
#             merged_flags[j] = True
#
#         max_dist = -1;
#         final_p1, final_p2 = None, None
#         for p_start_idx in range(len(line_group_points)):
#             for p_end_idx in range(p_start_idx + 1, len(line_group_points)):
#                 dist = point_distance(line_group_points[p_start_idx], line_group_points[p_end_idx])
#                 if dist > max_dist:
#                     max_dist = dist;
#                     final_p1 = line_group_points[p_start_idx];
#                     final_p2 = line_group_points[p_end_idx]
#         if final_p1 and final_p2:
#             merged_lines_final.append(
#                 {"points": (final_p1[0], final_p1[1], final_p2[0], final_p2[1]), "type": current_type})
#     return merged_lines_final
#
#
# def group_lines_into_dashed(lines):
#     """Analyzes lines to find and group those forming a dashed pattern."""
#     if len(lines) < DASHED_GROUP_MIN_LINES:
#         return [], lines
#
#     lines.sort(key=lambda l: l['angle'])  # Sort by angle to group collinear candidates
#
#     grouped_flags = [False] * len(lines)
#     dashed_groups = []
#
#     for i in range(len(lines)):
#         if grouped_flags[i]: continue
#
#         seed_line = lines[i]
#         potential_group = [seed_line]
#
#         # Find all other lines that are collinear and have similar length
#         for j in range(i + 1, len(lines)):
#             if grouped_flags[j]: continue
#             candidate_line = lines[j]
#
#             angle_diff = abs(seed_line['angle'] - candidate_line['angle'])
#             if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES: continue
#
#             length_ratio = seed_line['length'] / candidate_line['length']
#             if not (1 - DASHED_LENGTH_TOLERANCE < length_ratio < 1 + DASHED_LENGTH_TOLERANCE): continue
#
#             potential_group.append(candidate_line)
#
#         if len(potential_group) < DASHED_GROUP_MIN_LINES: continue
#
#         # Sort the potential group by position to check gaps
#         midpoints = [
#             ((l['endpoints'][0][0] + l['endpoints'][1][0]) / 2, (l['endpoints'][0][1] + l['endpoints'][1][1]) / 2) for l
#             in potential_group]
#         sort_key = midpoints[0][0] if abs(seed_line['angle']) < 45 or abs(seed_line['angle']) > 135 else midpoints[0][1]
#         potential_group.sort(
#             key=lambda l: (l['endpoints'][0][0] + l['endpoints'][1][0]) / 2 if abs(l['angle']) < 45 or abs(
#                 l['angle']) > 135 else (l['endpoints'][0][1] + l['endpoints'][1][1]) / 2)
#
#         # Check for regular gaps
#         gaps = []
#         for k in range(len(potential_group) - 1):
#             dist = point_distance(potential_group[k]['endpoints'][1], potential_group[k + 1]['endpoints'][0])
#             gaps.append(dist)
#
#         if not gaps: continue
#
#         avg_gap = sum(gaps) / len(gaps)
#         avg_len = sum(l['length'] for l in potential_group) / len(potential_group)
#
#         if avg_gap > avg_len * DASHED_MAX_GAP_FACTOR: continue  # Gaps are too large
#
#         gaps_are_regular = all(1 - DASHED_GAP_TOLERANCE < g / avg_gap < 1 + DASHED_GAP_TOLERANCE for g in gaps)
#
#         if gaps_are_regular:
#             # Mark all lines in the group as processed
#             for line_in_group in potential_group:
#                 for k in range(len(lines)):
#                     if lines[k] == line_in_group:
#                         grouped_flags[k] = True
#                         break
#
#             # Find the two extreme endpoints
#             all_points = [p for l in potential_group for p in l['endpoints']]
#             max_dist = -1;
#             final_p1, final_p2 = None, None
#             for p_start_idx in range(len(all_points)):
#                 for p_end_idx in range(p_start_idx + 1, len(all_points)):
#                     dist = point_distance(all_points[p_start_idx], all_points[p_end_idx])
#                     if dist > max_dist:
#                         max_dist = dist;
#                         final_p1 = all_points[p_start_idx];
#                         final_p2 = all_points[p_end_idx]
#             dashed_groups.append({"points": (final_p1[0], final_p1[1], final_p2[0], final_p2[1]), "type": "dashed"})
#
#     remaining_lines = [lines[i] for i in range(len(lines)) if not grouped_flags[i]]
#     return dashed_groups, remaining_lines
#
#
# # ---------- Main Analysis Function ----------
#
# def analyze_plan_final(file_path, out_path="out_final_dashed.jpg"):
#     gray_img, original_img, binary_img = convert_to_image(file_path, dpi=PDF_RENDER_DPI)
#     if gray_img is None: return {"error": "Failed to load."}
#
#     final_lines_to_draw = []
#
#     # 1. Process colorful lines (highest priority)
#     img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
#     mask_yellow = cv2.inRange(img_hsv, YELLOW_LOWER, UPPER_YELLOW)
#     yellow_lines_raw = cv2.HoughLinesP(mask_yellow, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
#     if yellow_lines_raw is not None:
#         for line in yellow_lines_raw:
#             final_lines_to_draw.append({"points": line[0], "type": "yellow"})
#
#     # 2. Prepare for grayscale processing
#     mask_inv = cv2.bitwise_not(cv2.dilate(mask_yellow, np.ones((15, 15), np.uint8)))
#     gray_masked = cv2.bitwise_and(gray_img, gray_img, mask=mask_inv)
#
#     # 3. Detect all grayscale lines with LSD
#     lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
#     lines_lsd, _, _, _ = lsd.detect(gray_masked)
#
#     # 4. Classify lines with the robust thickness method and separate by type
#     lines_by_type = defaultdict(list)
#     if lines_lsd is not None:
#         for i, seg in enumerate(lines_lsd):
#             p = seg[0];
#             length = math.hypot(p[2] - p[0], p[3] - p[1])
#             if length < MIN_LINE_LEN_PX: continue
#
#             line_obj = {
#                 'endpoints': ((int(p[0]), int(p[1])), (int(p[2]), int(p[3]))),
#                 'angle': math.degrees(math.atan2(p[3] - p[1], p[2] - p[0])),
#                 'length': length
#             }
#             thickness = get_robust_thickness(line_obj, binary_img)
#             line_obj['type'] = classify_thickness(thickness)
#             lines_by_type[line_obj['type']].append(line_obj)
#
#     # 5. Detect dashed lines from the 'thin' candidates
#     dashed_groups, remaining_thin_lines = group_lines_into_dashed(lines_by_type['thin'])
#     final_lines_to_draw.extend(dashed_groups)
#
#     # 6. Merge remaining solid lines
#     final_lines_to_draw.extend(merge_lines(remaining_thin_lines))
#     final_lines_to_draw.extend(merge_lines(lines_by_type['normal']))
#     final_lines_to_draw.extend(merge_lines(lines_by_type['thick']))
#
#     # 7. Draw the final results
#     output_img = original_img.copy()
#     for line_info in final_lines_to_draw:
#         x1, y1, x2, y2 = map(int, line_info["points"])
#         color = COLOR_MAP.get(line_info["type"])
#         cv2.line(output_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
#
#     cv2.imwrite(out_path, output_img);
#     print(f"Analysis complete. Drew {len(final_lines_to_draw)} final lines to {out_path}")
#     return {"count": len(final_lines_to_draw)}


# import fitz  # PyMuPDF
# import cv2
# import numpy as np
# import math
# from collections import defaultdict
#
# # ---------- Configuration & Constants ----------
#
# PDF_RENDER_DPI = 300
# MIN_LINE_LEN_PX = 15
#
# # Classification thresholds
# THIN_NORMAL_THRESHOLD = 2.8
# NORMAL_THICK_THRESHOLD = 4.5
#
# # Merging and Grouping parameters
# MERGE_DISTANCE_TOLERANCE_PIXELS = 5.0
# MERGE_ANGLE_TOLERANCE_DEGREES = 2.0
#
# # Refined parameters for dashed line detection
# DASHED_GROUP_MIN_LINES = 3
# DASHED_LENGTH_TOLERANCE = 0.5  # Segments can be +/- 50% of the average length
# DASHED_GAP_TOLERANCE = 0.5  # Gaps can be +/- 50% of the average gap
# DASHED_MAX_GAP_FACTOR = 3  # A gap cannot be more than 3x the average dash length
#
# # Color map
# COLOR_MAP = {
#     "thin": (0, 255, 0), "normal": (255, 0, 0), "thick": (0, 0, 255),
#     "yellow": (0, 255, 255), "dashed": (128, 0, 128),
# }
#
# # HSV color range
# YELLOW_LOWER = np.array([15, 50, 150]);
# UPPER_YELLOW = np.array([45, 255, 255])
#
#
# # ---------- Utility Functions ----------
# # convert_to_image, get_robust_thickness, classify_thickness, point_distance remain the same.
# def convert_to_image(file_path, dpi=300):
#     try:
#         if file_path.lower().endswith('.pdf'):
#             doc = fitz.open(file_path)
#             page = doc[0]
#             zoom = dpi / 72;
#             matrix = fitz.Matrix(zoom, zoom)
#             pix = page.get_pixmap(matrix=matrix, alpha=False)
#             img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             return gray, img, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#                                                     21, 10)
#         img = cv2.imread(file_path)
#         if img is None: raise FileNotFoundError(f"File not found: {file_path}")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return gray, img, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21,
#                                                 10)
#     except Exception as e:
#         print(f"Error converting file '{file_path}': {e}");
#         return None, None, None
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
#
# def point_distance(p1, p2):
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
#
#
# # ---------- Core Logic Functions ----------
# def merge_lines(lines_to_merge):
#     if not lines_to_merge: return []
#     merged_flags = [False] * len(lines_to_merge)
#     merged_lines_final = []
#     for i in range(len(lines_to_merge)):
#         if merged_flags[i]: continue
#         base_line = lines_to_merge[i];
#         current_type = base_line['type']
#         line_group_points = list(base_line['endpoints'])
#         merged_flags[i] = True
#         for j in range(i + 1, len(lines_to_merge)):
#             if merged_flags[j]: continue
#             candidate_line = lines_to_merge[j]
#             if candidate_line['type'] != current_type: continue
#             angle_diff = abs(base_line['angle'] - candidate_line['angle'])
#             if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES: continue
#             p1, p2 = base_line['endpoints'];
#             q1, q2 = candidate_line['endpoints']
#             if min(point_distance(p1, q1), point_distance(p1, q2), point_distance(p2, q1),
#                    point_distance(p2, q2)) > MERGE_DISTANCE_TOLERANCE_PIXELS: continue
#             line_group_points.extend(candidate_line['endpoints']);
#             merged_flags[j] = True
#         max_dist = -1;
#         final_p1, final_p2 = None, None
#         for p_start_idx in range(len(line_group_points)):
#             for p_end_idx in range(p_start_idx + 1, len(line_group_points)):
#                 dist = point_distance(line_group_points[p_start_idx], line_group_points[p_end_idx])
#                 if dist > max_dist: max_dist = dist; final_p1 = line_group_points[p_start_idx]; final_p2 = \
#                 line_group_points[p_end_idx]
#         if final_p1 and final_p2:
#             merged_lines_final.append(
#                 {"points": (final_p1[0], final_p1[1], final_p2[0], final_p2[1]), "type": current_type})
#     return merged_lines_final
#
#
# def group_lines_into_dashed(lines):
#     """Improved method to find and group segments that form a dashed line."""
#     if len(lines) < DASHED_GROUP_MIN_LINES:
#         return [], lines
#
#     line_indices = {id(line): i for i, line in enumerate(lines)}
#     grouped_flags = [False] * len(lines)
#     dashed_lines = []
#
#     for i in range(len(lines)):
#         if grouped_flags[i]: continue
#
#         seed_line = lines[i]
#
#         # 1. Form a potential group of collinear lines with similar length
#         group = [seed_line]
#         avg_len = seed_line['length']
#         for j in range(i + 1, len(lines)):
#             if grouped_flags[j]: continue
#             candidate = lines[j]
#
#             angle_diff = abs(seed_line['angle'] - candidate['angle'])
#             if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES: continue
#             if abs(candidate['length'] - avg_len) / avg_len > DASHED_LENGTH_TOLERANCE: continue
#
#             group.append(candidate)
#
#         if len(group) < DASHED_GROUP_MIN_LINES: continue
#
#         # 2. Sort the group members along their common axis for proper gap analysis
#         ref_point = np.array(group[0]['endpoints'][0])
#         ref_vec = np.array(group[0]['endpoints'][1]) - ref_point
#         group.sort(key=lambda l: np.dot(np.array(l['endpoints'][0]) - ref_point, ref_vec))
#
#         # 3. Analyze the gaps between the sorted segments
#         gaps = []
#         for k in range(len(group) - 1):
#             line_a, line_b = group[k], group[k + 1]
#             min_dist = min(point_distance(p_a, p_b) for p_a in line_a['endpoints'] for p_b in line_b['endpoints'])
#             gaps.append(min_dist)
#
#         if not gaps: continue
#
#         # 4. Check if gaps are regular and not too large
#         avg_gap = sum(gaps) / len(gaps)
#         avg_len = sum(l['length'] for l in group) / len(group)
#         if avg_gap == 0 or avg_gap > avg_len * DASHED_MAX_GAP_FACTOR: continue
#
#         is_regular = all(abs(g - avg_gap) / avg_gap < DASHED_GAP_TOLERANCE for g in gaps)
#
#         # 5. If it's a valid dashed line, create the final representation
#         if is_regular:
#             for line_in_group in group:
#                 grouped_flags[line_indices[id(line_in_group)]] = True
#
#             all_points = [p for l in group for p in l['endpoints']]
#             max_dist = -1;
#             final_p1, final_p2 = None, None
#             for p_start_idx in range(len(all_points)):
#                 for p_end_idx in range(p_start_idx + 1, len(all_points)):
#                     dist = point_distance(all_points[p_start_idx], all_points[p_end_idx])
#                     if dist > max_dist: max_dist = dist; final_p1 = all_points[p_start_idx]; final_p2 = all_points[
#                         p_end_idx]
#             dashed_lines.append({"points": (final_p1[0], final_p1[1], final_p2[0], final_p2[1]), "type": "dashed"})
#
#     remaining_lines = [lines[i] for i in range(len(lines)) if not grouped_flags[i]]
#     return dashed_lines, remaining_lines
#
#
# # ---------- Main Analysis Function ----------
# def analyze_plan_final(file_path, out_path="out_final_dashed.jpg"):
#     gray_img, original_img, binary_img = convert_to_image(file_path, dpi=PDF_RENDER_DPI)
#     if gray_img is None: return {"error": "Failed to load."}
#
#     final_lines_to_draw = []
#
#     # 1. Process colorful lines
#     img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
#     mask_yellow = cv2.inRange(img_hsv, YELLOW_LOWER, UPPER_YELLOW)
#     yellow_lines_raw = cv2.HoughLinesP(mask_yellow, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
#     if yellow_lines_raw is not None:
#         for line in yellow_lines_raw: final_lines_to_draw.append({"points": line[0], "type": "yellow"})
#
#     # 2. Prepare for grayscale processing
#     mask_inv = cv2.bitwise_not(cv2.dilate(mask_yellow, np.ones((15, 15), np.uint8)))
#     gray_masked = cv2.bitwise_and(gray_img, gray_img, mask=mask_inv)
#     binary_masked = cv2.bitwise_and(binary_img, binary_img, mask=mask_inv)
#
#     # 3. Detect all grayscale lines and classify with robust thickness
#     lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
#     lines_lsd, _, _, _ = lsd.detect(gray_masked)
#     lines_by_type = defaultdict(list)
#     if lines_lsd is not None:
#         for seg in lines_lsd:
#             p = seg[0];
#             length = math.hypot(p[2] - p[0], p[3] - p[1])
#             if length < MIN_LINE_LEN_PX: continue
#             line_obj = {'endpoints': ((int(p[0]), int(p[1])), (int(p[2]), int(p[3]))),
#                         'angle': math.degrees(math.atan2(p[3] - p[1], p[2] - p[0])), 'length': length}
#             thickness = get_robust_thickness(line_obj, binary_masked)
#             line_obj['type'] = classify_thickness(thickness)
#             lines_by_type[line_obj['type']].append(line_obj)
#
#     # 4. Detect dashed lines from the 'thin' candidates
#     dashed_groups, remaining_thin_lines = group_lines_into_dashed(lines_by_type['thin'])
#     final_lines_to_draw.extend(dashed_groups)
#
#     # 5. Merge remaining solid lines
#     final_lines_to_draw.extend(merge_lines(remaining_thin_lines))
#     final_lines_to_draw.extend(merge_lines(lines_by_type['normal']))
#     final_lines_to_draw.extend(merge_lines(lines_by_type['thick']))
#
#     # 6. Draw the final results
#     output_img = original_img.copy()
#     for line_info in final_lines_to_draw:
#         x1, y1, x2, y2 = map(int, line_info["points"])
#         color = COLOR_MAP.get(line_info["type"])
#         cv2.line(output_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
#
#     cv2.imwrite(out_path, output_img);
#     print(f"Analysis complete. Drew {len(final_lines_to_draw)} final lines to {out_path}")
#     return {"count": len(final_lines_to_draw)}
#
#
# # # ---------- Execution ----------
# # if __name__ == "__main__":
# #     analyze_plan_final("6.pdf")
#
# # ---------- Execution ----------
# if __name__ == "__main__":
#     analyze_plan_final(file_path)
import cv2
import numpy as np
import fitz  # PyMuPDF
import math
from collections import defaultdict
from itertools import chain
# ---------- Configuration & Constants ----------
PDF_RENDER_DPI = 300
MIN_LINE_LEN_PX = 30  # Increased to aggressively filter out text fragments

# Classification thresholds
THIN_NORMAL_THRESHOLD = 2.8
NORMAL_THICK_THRESHOLD = 4.5

# Merging and Grouping parameters
MERGE_ANGLE_TOLERANCE_DEGREES = 3.0
MERGE_DISTANCE_TOLERANCE_PIXELS = 10.0

DASHED_GROUP_MIN_LINES = 3
DASHED_MIN_GAP_THRESHOLD = 5  # Gaps must be >5px to be considered dashed
DASHED_GAP_TOLERANCE = 0.6
DASHED_LENGTH_TOLERANCE = 0.6

# Color map
COLOR_MAP = {
    "thin": (0, 255, 0), "normal": (255, 0, 0), "thick": (0, 0, 255),
    "yellow": (0, 255, 255), "dashed": (128, 0, 128),
}

# HSV color range
YELLOW_LOWER = np.array([15, 50, 150]);
UPPER_YELLOW = np.array([45, 255, 255])


# ---------- Utility Functions ----------
def convert_to_image(file_path, dpi=300):
    """Loads a PDF or image and returns color, grayscale, and binary versions."""
    try:
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            page = doc[0]
            zoom = dpi / 72;
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
            return gray, img, binary
        img = cv2.imread(file_path)
        if img is None: raise FileNotFoundError(f"File not found: {file_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        return gray, img, binary
    except Exception as e:
        print(f"Error converting file '{file_path}': {e}");
        return None, None, None


def get_robust_thickness(line, binary_image):
    """Measures the thickness of a line on the original binary image using its centerline."""
    p1, p2 = line['endpoints'];
    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    perp_angle_rad = angle_rad + math.pi / 2
    mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    thickness_samples = []
    # Sample points across the line's perpendicular to measure width
    for i in range(-10, 11):  # Sample 21 points
        thickness = 0
        check_x, check_y = mid_x + i * math.cos(angle_rad), mid_y + i * math.sin(angle_rad)
        for j in range(-15, 15):
            x = int(check_x + j * math.cos(perp_angle_rad))
            y = int(check_y + j * math.sin(perp_angle_rad))
            if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1] and binary_image[y, x] != 0:
                thickness += 1
        thickness_samples.append(thickness)
    return max(1, np.mean(thickness_samples))  # Use the average of samples for stability


def classify_thickness(width):
    if width <= THIN_NORMAL_THRESHOLD:
        return "thin"
    elif width <= NORMAL_THICK_THRESHOLD:
        return "normal"
    else:
        return "thick"


def point_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_line_angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


# ---------- Core Logic Functions ----------
def merge_collinear_lines(lines):
    """Iteratively merges the best pair of collinear solid lines until no more merges are possible."""
    if not lines: return []
    while True:
        best_merge = None;
        max_len = -1
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1, line2 = lines[i], lines[j]
                angle_diff = abs(line1['angle'] - line2['angle'])
                if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES: continue
                p1, p2 = line1['endpoints'];
                q1, q2 = line2['endpoints']
                dist = min(point_distance(p1, q1), point_distance(p1, q2), point_distance(p2, q1),
                           point_distance(p2, q2))
                if dist > MERGE_DISTANCE_TOLERANCE_PIXELS: continue
                all_points = [p1, p2, q1, q2]
                max_dist = 0
                for p_start in all_points:
                    for p_end in all_points: max_dist = max(max_dist, point_distance(p_start, p_end))
                if max_dist > max_len: max_len = max_dist; best_merge = (i, j, all_points)
        if best_merge is None: break
        i, j, all_points = best_merge
        max_dist = -1;
        final_p1, final_p2 = None, None
        for p_start_idx in range(len(all_points)):
            for p_end_idx in range(p_start_idx + 1, len(all_points)):
                dist = point_distance(all_points[p_start_idx], all_points[p_end_idx])
                if dist > max_dist: max_dist = dist; final_p1 = all_points[p_start_idx]; final_p2 = all_points[
                    p_end_idx]
        base_line = lines[i] if lines[i]['length'] > lines[j]['length'] else lines[j]
        new_line = {'endpoints': (final_p1, final_p2), 'angle': get_line_angle(final_p1, final_p2), 'length': max_dist,
                    'type': base_line['type']}
        lines.pop(j if j > i else i);
        lines.pop(i if j > i else j)
        lines.append(new_line)
    return lines


def group_lines_into_dashed(lines):
    if len(lines) < DASHED_GROUP_MIN_LINES: return [], lines
    line_indices = {id(line): i for i, line in enumerate(lines)};
    grouped_flags = [False] * len(lines)
    dashed_lines = []
    for i in range(len(lines)):
        if grouped_flags[i]: continue
        seed_line = lines[i]
        group = [seed_line]
        for j in range(i + 1, len(lines)):
            if grouped_flags[j]: continue
            candidate = lines[j]
            angle_diff = abs(seed_line['angle'] - candidate['angle'])
            if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES: continue
            if abs(candidate['length'] - seed_line['length']) / seed_line['length'] > DASHED_LENGTH_TOLERANCE: continue
            group.append(candidate)
        if len(group) < DASHED_GROUP_MIN_LINES: continue
        ref_point = np.array(group[0]['endpoints'][0]);
        ref_vec = np.array(group[0]['endpoints'][1]) - ref_point
        group.sort(key=lambda l: np.dot(np.array(l['endpoints'][0]) - ref_point, ref_vec))
        gaps = [point_distance(group[k]['endpoints'][1], group[k + 1]['endpoints'][0]) for k in range(len(group) - 1)]
        if not gaps: continue
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap < DASHED_MIN_GAP_THRESHOLD: continue
        avg_len = sum(l['length'] for l in group) / len(group)
        if avg_gap > avg_len * 3: continue
        is_regular = all(abs(g - avg_gap) / avg_gap < DASHED_GAP_TOLERANCE for g in gaps)
        if is_regular:
            for line_in_group in group: grouped_flags[line_indices[id(line_in_group)]] = True
            all_points = [p for l in group for p in l['endpoints']]
            max_dist = -1;
            final_p1, final_p2 = None, None
            for p_start_idx in range(len(all_points)):
                for p_end_idx in range(p_start_idx + 1, len(all_points)):
                    dist = point_distance(all_points[p_start_idx], all_points[p_end_idx])
                    if dist > max_dist: max_dist = dist; final_p1 = all_points[p_start_idx]; final_p2 = all_points[
                        p_end_idx]
            # **FIX**: Ensure the returned line object has the standard 'endpoints' structure
            dashed_lines.append({"endpoints": (final_p1, final_p2), "type": "dashed"})
    remaining_lines = [lines[i] for i in range(len(lines)) if not grouped_flags[i]]
    return dashed_lines, remaining_lines


# ---------- Main Analysis Function ----------
def analyze_plan_with_skeleton(file_path, out_path="out_skeleton_final.jpg"):
    _, original_img, binary_img = convert_to_image(file_path, dpi=PDF_RENDER_DPI)
    if original_img is None: return {"error": "Failed to load."}

    final_lines = []

    # 1. Process colorful lines
    img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(img_hsv, YELLOW_LOWER, UPPER_YELLOW)
    yellow_lines_raw = cv2.HoughLinesP(mask_yellow, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
    if yellow_lines_raw is not None:
        for line in yellow_lines_raw:
            p = line[0]
            # **FIX**: Create yellow lines with the standard 'endpoints' structure
            final_lines.append({"endpoints": ((p[0], p[1]), (p[2], p[3])), "type": "yellow"})

    # 2. Use a skeletonized image for grayscale line detection
    mask_inv = cv2.bitwise_not(cv2.dilate(mask_yellow, np.ones((15, 15), np.uint8)))
    binary_masked = cv2.bitwise_and(binary_img, binary_img, mask=mask_inv)
    skeleton = cv2.ximgproc.thinning(binary_masked)

    # 3. Detect centerlines from the skeleton
    detected_centerlines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=20, minLineLength=MIN_LINE_LEN_PX,
                                           maxLineGap=10)

    # 4. Classify lines based on robust thickness
    lines_by_type = defaultdict(list)
    if detected_centerlines is not None:
        for seg in detected_centerlines:
            p = seg[0];
            length = math.hypot(p[2] - p[0], p[3] - p[1])
            if length < MIN_LINE_LEN_PX: continue
            p1, p2 = (int(p[0]), int(p[1])), (int(p[2]), int(p[3]))
            # **FIX**: All lines are created with the standard 'endpoints' structure from the start
            line_obj = {'endpoints': (p1, p2), 'angle': get_line_angle(p1, p2), 'length': length}
            thickness = get_robust_thickness(line_obj, binary_img)
            line_obj['type'] = classify_thickness(thickness)
            lines_by_type[line_obj['type']].append(line_obj)

    # 5. Group and merge lines
    dashed_groups, remaining_thin = group_lines_into_dashed(list(chain(*lines_by_type.values())))
    final_lines.extend(dashed_groups)
    merged_thin = merge_collinear_lines(remaining_thin)
    merged_normal = merge_collinear_lines(lines_by_type['normal'])
    merged_thick = merge_collinear_lines(lines_by_type['thick'])
    final_lines.extend(merged_thin + merged_normal + merged_thick)

    # 6. Draw final results
    output_img = original_img.copy()
    for line_info in final_lines:
        # **FIX**: This drawing loop is now simple and safe because all line objects are consistent
        p1, p2 = line_info["endpoints"]
        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)
        color = COLOR_MAP.get(line_info["type"], (255, 255, 255))
        cv2.line(output_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    cv2.imwrite(out_path, output_img)
    print(f"Analysis complete. Drew {len(final_lines)} final lines to {out_path}")
    return {"count": len(final_lines)}


# ---------- Execution ----------
if __name__ == "__main__":
    analyze_plan_with_skeleton(file_path, "ex1.jpg")