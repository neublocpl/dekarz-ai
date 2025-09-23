file_path = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow/6.pdf"

import fitz  # PyMuPDF
import cv2
import numpy as np
import math
from collections import defaultdict

# ---------- Configuration & Constants ----------

PDF_RENDER_DPI = 300
MIN_LINE_LEN_PX = 15

# Classification thresholds
THIN_NORMAL_THRESHOLD = 2.8
NORMAL_THICK_THRESHOLD = 4.5

# Merging and Grouping parameters
MERGE_DISTANCE_TOLERANCE_PIXELS = 5.0
MERGE_ANGLE_TOLERANCE_DEGREES = 2.0

# Refined parameters for dashed line detection
DASHED_GROUP_MIN_LINES = 3
DASHED_LENGTH_TOLERANCE = 0.5  # Segments can be +/- 50% of the average length
DASHED_GAP_TOLERANCE = 0.5  # Gaps can be +/- 50% of the average gap
DASHED_MAX_GAP_FACTOR = 3  # A gap cannot be more than 3x the average dash length

# Color map
COLOR_MAP = {
    "thin": (0, 255, 0), "normal": (255, 0, 0), "thick": (0, 0, 255),
    "yellow": (0, 255, 255), "dashed": (128, 0, 128),
}

# HSV color range
YELLOW_LOWER = np.array([15, 50, 150]);
UPPER_YELLOW = np.array([45, 255, 255])


# ---------- Utility Functions ----------
# convert_to_image, get_robust_thickness, classify_thickness, point_distance remain the same.
def convert_to_image(file_path, dpi=300):
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
            return gray, img, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                    21, 10)
        img = cv2.imread(file_path)
        if img is None: raise FileNotFoundError(f"File not found: {file_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray, img, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21,
                                                10)
    except Exception as e:
        print(f"Error converting file '{file_path}': {e}");
        return None, None, None


def get_robust_thickness(line, binary_image):
    p1, p2 = line['endpoints'];
    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    perp_angle_rad = angle_rad + math.pi / 2
    mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    thickness = 0;
    max_check = 15
    for i in range(max_check):
        x, y = int(mid_x + i * math.cos(perp_angle_rad)), int(mid_y + i * math.sin(perp_angle_rad))
        if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1] and binary_image[y, x] != 0:
            thickness += 1
        else:
            break
    for i in range(1, max_check):
        x, y = int(mid_x - i * math.cos(perp_angle_rad)), int(mid_y - i * math.sin(perp_angle_rad))
        if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1] and binary_image[y, x] != 0:
            thickness += 1
        else:
            break
    return max(1, thickness)


def classify_thickness(width):
    if width <= THIN_NORMAL_THRESHOLD:
        return "thin"
    elif width <= NORMAL_THICK_THRESHOLD:
        return "normal"
    else:
        return "thick"


def point_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


# ---------- Core Logic Functions ----------
def merge_lines(lines_to_merge):
    if not lines_to_merge: return []
    merged_flags = [False] * len(lines_to_merge)
    merged_lines_final = []
    for i in range(len(lines_to_merge)):
        if merged_flags[i]: continue
        base_line = lines_to_merge[i];
        current_type = base_line['type']
        line_group_points = list(base_line['endpoints'])
        merged_flags[i] = True
        for j in range(i + 1, len(lines_to_merge)):
            if merged_flags[j]: continue
            candidate_line = lines_to_merge[j]
            if candidate_line['type'] != current_type: continue
            angle_diff = abs(base_line['angle'] - candidate_line['angle'])
            if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES: continue
            p1, p2 = base_line['endpoints'];
            q1, q2 = candidate_line['endpoints']
            if min(point_distance(p1, q1), point_distance(p1, q2), point_distance(p2, q1),
                   point_distance(p2, q2)) > MERGE_DISTANCE_TOLERANCE_PIXELS: continue
            line_group_points.extend(candidate_line['endpoints']);
            merged_flags[j] = True
        max_dist = -1;
        final_p1, final_p2 = None, None
        for p_start_idx in range(len(line_group_points)):
            for p_end_idx in range(p_start_idx + 1, len(line_group_points)):
                dist = point_distance(line_group_points[p_start_idx], line_group_points[p_end_idx])
                if dist > max_dist: max_dist = dist; final_p1 = line_group_points[p_start_idx]; final_p2 = \
                line_group_points[p_end_idx]
        if final_p1 and final_p2:
            merged_lines_final.append(
                {"points": (final_p1[0], final_p1[1], final_p2[0], final_p2[1]), "type": current_type})
    return merged_lines_final


def group_lines_into_dashed(lines):
    """Improved method to find and group segments that form a dashed line."""
    if len(lines) < DASHED_GROUP_MIN_LINES:
        return [], lines

    line_indices = {id(line): i for i, line in enumerate(lines)}
    grouped_flags = [False] * len(lines)
    dashed_lines = []

    for i in range(len(lines)):
        if grouped_flags[i]: continue

        seed_line = lines[i]

        # 1. Form a potential group of collinear lines with similar length
        group = [seed_line]
        avg_len = seed_line['length']
        for j in range(i + 1, len(lines)):
            if grouped_flags[j]: continue
            candidate = lines[j]

            angle_diff = abs(seed_line['angle'] - candidate['angle'])
            if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES: continue
            if abs(candidate['length'] - avg_len) / avg_len > DASHED_LENGTH_TOLERANCE: continue

            group.append(candidate)

        if len(group) < DASHED_GROUP_MIN_LINES: continue

        # 2. Sort the group members along their common axis for proper gap analysis
        ref_point = np.array(group[0]['endpoints'][0])
        ref_vec = np.array(group[0]['endpoints'][1]) - ref_point
        group.sort(key=lambda l: np.dot(np.array(l['endpoints'][0]) - ref_point, ref_vec))

        # 3. Analyze the gaps between the sorted segments
        gaps = []
        for k in range(len(group) - 1):
            line_a, line_b = group[k], group[k + 1]
            min_dist = min(point_distance(p_a, p_b) for p_a in line_a['endpoints'] for p_b in line_b['endpoints'])
            gaps.append(min_dist)

        if not gaps: continue

        # 4. Check if gaps are regular and not too large
        avg_gap = sum(gaps) / len(gaps)
        avg_len = sum(l['length'] for l in group) / len(group)
        if avg_gap == 0 or avg_gap > avg_len * DASHED_MAX_GAP_FACTOR: continue

        is_regular = all(abs(g - avg_gap) / avg_gap < DASHED_GAP_TOLERANCE for g in gaps)

        # 5. If it's a valid dashed line, create the final representation
        if is_regular:
            for line_in_group in group:
                grouped_flags[line_indices[id(line_in_group)]] = True

            all_points = [p for l in group for p in l['endpoints']]
            max_dist = -1;
            final_p1, final_p2 = None, None
            for p_start_idx in range(len(all_points)):
                for p_end_idx in range(p_start_idx + 1, len(all_points)):
                    dist = point_distance(all_points[p_start_idx], all_points[p_end_idx])
                    if dist > max_dist: max_dist = dist; final_p1 = all_points[p_start_idx]; final_p2 = all_points[
                        p_end_idx]
            dashed_lines.append({"points": (final_p1[0], final_p1[1], final_p2[0], final_p2[1]), "type": "dashed"})

    remaining_lines = [lines[i] for i in range(len(lines)) if not grouped_flags[i]]
    return dashed_lines, remaining_lines


# ---------- Main Analysis Function ----------
def analyze_plan_final(file_path, out_path="out_final_dashed.jpg"):
    gray_img, original_img, binary_img = convert_to_image(file_path, dpi=PDF_RENDER_DPI)
    if gray_img is None: return {"error": "Failed to load."}

    final_lines_to_draw = []

    # 1. Process colorful lines
    img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(img_hsv, YELLOW_LOWER, UPPER_YELLOW)
    yellow_lines_raw = cv2.HoughLinesP(mask_yellow, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
    if yellow_lines_raw is not None:
        for line in yellow_lines_raw: final_lines_to_draw.append({"points": line[0], "type": "yellow"})

    # 2. Prepare for grayscale processing
    mask_inv = cv2.bitwise_not(cv2.dilate(mask_yellow, np.ones((15, 15), np.uint8)))
    gray_masked = cv2.bitwise_and(gray_img, gray_img, mask=mask_inv)
    binary_masked = cv2.bitwise_and(binary_img, binary_img, mask=mask_inv)

    # 3. Detect all grayscale lines and classify with robust thickness
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines_lsd, _, _, _ = lsd.detect(gray_masked)
    lines_by_type = defaultdict(list)
    if lines_lsd is not None:
        for seg in lines_lsd:
            p = seg[0];
            length = math.hypot(p[2] - p[0], p[3] - p[1])
            if length < MIN_LINE_LEN_PX: continue
            line_obj = {'endpoints': ((int(p[0]), int(p[1])), (int(p[2]), int(p[3]))),
                        'angle': math.degrees(math.atan2(p[3] - p[1], p[2] - p[0])), 'length': length}
            thickness = get_robust_thickness(line_obj, binary_masked)
            line_obj['type'] = classify_thickness(thickness)
            lines_by_type[line_obj['type']].append(line_obj)

    # 4. Detect dashed lines from the 'thin' candidates
    dashed_groups, remaining_thin_lines = group_lines_into_dashed(lines_by_type['thin'])
    final_lines_to_draw.extend(dashed_groups)

    # 5. Merge remaining solid lines
    final_lines_to_draw.extend(merge_lines(remaining_thin_lines))
    final_lines_to_draw.extend(merge_lines(lines_by_type['normal']))
    final_lines_to_draw.extend(merge_lines(lines_by_type['thick']))

    # 6. Draw the final results
    output_img = original_img.copy()
    for line_info in final_lines_to_draw:
        x1, y1, x2, y2 = map(int, line_info["points"])
        color = COLOR_MAP.get(line_info["type"])
        cv2.line(output_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    cv2.imwrite(out_path, output_img);
    print(f"Analysis complete. Drew {len(final_lines_to_draw)} final lines to {out_path}")
    return {"count": len(final_lines_to_draw)}


methode = analyze_plan_final