
import cv2
import numpy as np
import fitz  # PyMuPDF
import math
from collections import defaultdict

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
    dashed_groups, remaining_thin = group_lines_into_dashed(lines_by_type['thin'])
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


methode = analyze_plan_with_skeleton