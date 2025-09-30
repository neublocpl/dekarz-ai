from typing import List, Tuple

import numpy as np

from .objects import Interval


def best_parallel_segment(
    points: List[Tuple[float, float]],
    versor: Tuple[float, float],
    return_extra: bool = False,
):
    """
    Given points and a versor (vx, vy), find the shortest line segment
    parallel to the versor that contains the orthogonal projections of all points,
    and minimize MSE of orthogonal distances to the infinite line.

    Args:
        points: list or array-like of (x, y) pairs.
        versor: (vx, vy) direction. Will be normalized if not unit.
        return_extra: if True returns (endpoint1, endpoint2, mse, details_dict)

    Returns:
        ( (X1, Y1), (X2, Y2), mse )  if return_extra is False
        otherwise ( (X1, Y1), (X2, Y2), mse, details_dict )
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an iterable of (x, y) pairs")

    vx, vy = float(versor[0]), float(versor[1])
    v_norm = np.hypot(vx, vy)
    if v_norm < 1e-12:
        raise ValueError("versor must be non-zero")

    # Unit direction vector
    v = np.array([vx, vy]) / v_norm

    # Unit normal vector (perpendicular). Pick (-vy, vx) then normalize (should already be unit).
    n = np.array([-v[1], v[0]])
    # n already unit if v unit, but ensure numerical stability:
    n = n / (np.linalg.norm(n) + 1e-15)

    # Projections onto normal and direction
    proj_n = pts @ n  # n·p_i, shape (N,)
    proj_v = pts @ v  # v·p_i, shape (N,)

    # Optimal offset b minimizing MSE of orthogonal distances
    b = proj_n.mean()

    # MSE (mean squared orthogonal distance)
    residuals = proj_n - b
    mse = float(np.mean(residuals**2))

    # Shortest interval along v that contains all projections (t_min, t_max)
    t_min, t_max = float(proj_v.min()), float(proj_v.max())

    # Endpoints in Cartesian coordinates
    p1 = v * t_min + n * b
    p2 = v * t_max + n * b
    endpoint1 = (float(p1[0]), float(p1[1]))
    endpoint2 = (float(p2[0]), float(p2[1]))

    from itertools import product

    y = 0
    for pa, pb in product(points, points):
        if np.linalg.norm(np.array(pa) - np.array(pb)) > y:
            y = np.linalg.norm(np.array(pa) - np.array(pb))
    x = np.linalg.norm(np.array(endpoint1) - np.array(endpoint2))
    # if y > 1.01 * x:
    #     print(f" > {x}, {y} / b-{b} v-{v} n-{n} tmin-{t_min} tmax-{t_max} p1-{p1} p2-{p2} ")
    #     print(points)

    #  > 140.76986882367822, 196.01020381602586 / b--815.844074075073 v-[0.00933293 0.99995645] n-[-0.99995645  0.00933293] tmin-1686.4212152533096 tmax-1827.1910840769879 p1-[ 831.54778768 1678.73355398] p2-[ 832.86158257 1819.4972919 ]
    # [
    # (803, 1679), (804, 1776), (809, 1681), (810, 1736), (841, 1778), (840, 1725),
    # (853, 1778), (853, 1728), (889, 1765), (889, 1716), (869, 1739), (869, 1693),
    # (857, 1743), (856, 1696), (764, 1724), (764, 1680), (936, 1774), (935, 1730),
    # (784, 1739), (784, 1779), (771, 1734), (772, 1773), (823, 1743), (823, 1778),
    # (904, 1740), (905, 1774), (794, 1715), (794, 1748), (856, 1748), (856, 1779),
    # (814, 1741), (814, 1713), (779, 1795), (779, 1820)]

    if return_extra:
        details = {
            "v_unit": tuple(v.tolist()),
            "n_unit": tuple(n.tolist()),
            "b": float(b),
            "t_min": t_min,
            "t_max": t_max,
            "proj_v": proj_v,  # numpy array of t_i
            "proj_n": proj_n,  # numpy array of n·p_i
        }
        return endpoint1, endpoint2, mse, details

    return endpoint1, endpoint2, mse


# import numpy as np

# def best_parallel_segment(intervals):
#     """
#     Find the best-fit line segment for given intervals using weighted linear regression.
#     Weights are proportional to interval length. The line minimizes weighted MSE and the
#     resulting segment contains all projections of endpoints.

#     Args:
#         intervals (list of tuple): list of intervals, each defined by two endpoints ((x1,y1),(x2,y2)).

#     Returns:
#         ((X1, Y1), (X2, Y2)): endpoints of the best-fit line segment.
#     """
#     # Flatten points and assign weights based on length of interval
#     pts = []
#     weights = []
#     for seg in intervals:
#         p1, p2 = np.array(seg[0]), np.array(seg[1])
#         length = np.linalg.norm(p2 - p1)
#         pts.extend([p1, p2])
#         weights.extend([length, length])
#     pts = np.array(pts)
#     weights = np.array(weights)

#     # Weighted mean of points
#     mean = np.average(pts, axis=0, weights=weights)

#     # Weighted covariance matrix
#     X = pts - mean
#     cov = (X * weights[:, None]).T @ X / np.sum(weights)

#     # Principal component analysis: direction = eigenvector with max eigenvalue
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     v = eigvecs[:, np.argmax(eigvals)]
#     v = v / np.linalg.norm(v)

#     # Orthogonal direction
#     n = np.array([-v[1], v[0]])

#     # Projection of mean onto n to find offset b
#     proj_n = pts @ n
#     weighted_mean_proj = np.average(proj_n, weights=weights)
#     base_point = mean + (weighted_mean_proj - mean @ n) * n

#     # Projection of all points onto v to get min/max
#     proj_v = pts @ v
#     min_proj, max_proj = np.min(proj_v), np.max(proj_v)

#     # Segment endpoints
#     p1 = tuple(base_point + v * min_proj)
#     p2 = tuple(base_point + v * max_proj)

#     return p1, p2, 0


# # Example usage
# intervals = [
#     ((0, 0), (1, 1)),
#     ((2, 2), (3, 2)),
#     ((4, 4), (6, 5))
# ]
# segment = best_fit_segment(intervals)
# print("Best-fit weighted line segment:", segment)
import numpy as np
from tqdm import tqdm

# def _are_collinear(p1, p2, p3, tol=1e-2, coll_thr=5):
#     """Check if points p1, p2, p3 are collinear within tolerance."""
#     area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
#     base = min(np.linalg.norm(np.array(p2) - np.array(p1)), np.linalg.norm(np.array(p2) - np.array(p3)), np.linalg.norm(np.array(p3) - np.array(p1)))
#     base = np.linalg.norm(np.array(p2) - np.array(p1))
#     # print(area, base, p1, p2, p3, area / (base + 1e-9), tol, tol * base)
#     return area / (base + 1e-9) < max(tol * base, coll_thr)


def _are_collinear(interval1, interval2, angle_tolerance_degrees=3, coll_thr=5):
    """
    Decide if two intervals are collinear within an angular tolerance.


    Args:
    interval1: ((x1,y1),(x2,y2)) first interval
    interval2: ((x3,y3),(x4,y4)) second interval
    angle_tolerance_degrees: maximum angle difference to consider collinear


    Returns:
    bool: True if intervals are collinear, False otherwise
    """
    p1, p2 = np.array(interval1[0]), np.array(interval1[1])
    q1, q2 = np.array(interval2[0]), np.array(interval2[1])

    v1 = p2 - p1
    v2 = q2 - q1

    # Guard against zero-length intervals
    if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
        return False

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Angle between vectors
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(abs(dot)))  # use abs to ignore orientation
    # print(angle)
    if angle > angle_tolerance_degrees:
        return False

    # v11 = p1 - q1
    # v12 = p1 - q2
    # v21 = p2 - q1
    # v22 = p2 - q2

    # sv1 = max([v11, v12], key = np.linalg.norm)
    # sv2 = max([v21, v22], key = np.linalg.norm)

    # sv1 = sv1 / np.linalg.norm(sv1)
    # sv2 = sv2 / np.linalg.norm(sv2)

    # dot11 = np.clip(np.dot(v1, sv1), -1.0, 1.0)
    # dot12 = np.clip(np.dot(v1, sv2), -1.0, 1.0)
    # dot21 = np.clip(np.dot(v2, sv1), -1.0, 1.0)
    # dot22 = np.clip(np.dot(v2, sv2), -1.0, 1.0)

    # angle1 = max([ np.degrees(np.arccos(abs(dot11))), np.degrees(np.arccos(abs(dot12)))])
    # angle2 = max([ np.degrees(np.arccos(abs(dot21))), np.degrees(np.arccos(abs(dot22)))])
    # # print(angle1, angle2, get_dist_to_interval(interval1, interval2), coll_thr, max([angle1, angle2]) <= angle_tolerance_degrees or get_dist_to_interval(interval1, interval2) < coll_thr)
    pts = np.array([q1 - p1, q2 - p1])
    vr = np.array([v1[1], -v1[0]])
    dists = pts @ vr

    # print(dists)

    # return max([angle1, angle2]) <= angle_tolerance_degrees or get_dist_to_interval(interval1, interval2) < coll_thr
    # print(get_dist_to_interval(interval1, interval2), coll_thr)
    return np.max(np.abs(dists)) < coll_thr


def _distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _min_distance_between_segments(seg1, seg2):
    # print(seg1, seg2)
    """Compute minimal distance between two line segments."""

    def point_line_distance(p, a, b):
        ap = np.array(p) - np.array(a)
        ab = np.array(b) - np.array(a)
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)
        t = max(0, min(1, t))
        proj = np.array(a) + t * ab
        return np.linalg.norm(np.array(p) - proj)

    a1, a2 = seg1
    b1, b2 = seg2

    # print(point_line_distance(a1, b1, b2),
    #     point_line_distance(a2, b1, b2),
    #     point_line_distance(b1, a1, a2),
    #     point_line_distance(b2, a1, a2))

    return min(
        point_line_distance(a1, b1, b2),
        point_line_distance(a2, b1, b2),
        point_line_distance(b1, a1, a2),
        point_line_distance(b2, a1, a2),
    )


def _lines_has_same_angle(
    line1: Interval,
    line2: Interval,
    scale=1000,
    coll_thr_scale_coeff=0.002,
    angle_tolerance_degrees=2,
):
    coll_thr = scale * coll_thr_scale_coeff
    # print(line1["angle_label"] == line2["angle_label"] , (_are_collinear(*line1["endpoints"], line2["endpoints"][0], coll_thr=coll_thr) , _are_collinear(*line1["endpoints"], line2["endpoints"][1], coll_thr=coll_thr)))
    # return line1["angle_label"] == line2["angle_label"] or
    return _are_collinear(
        line1.endpoints,
        line2.endpoints,
        angle_tolerance_degrees=angle_tolerance_degrees,
        coll_thr=coll_thr,
    )  # [0], coll_thr=coll_thr) and _are_collinear(*line1["endpoints"], line2["endpoints"][1], coll_thr=coll_thr))


def get_dist_to_interval(interval1, interval2):
    """
    Compute the minimal distance between two intervals (line segments).
    If they intersect, the distance is 0.


    Args:
    interval1: ((x1,y1),(x2,y2))
    interval2: ((x3,y3),(x4,y4))


    Returns:
    float: minimal distance between the two segments.
    """
    p1, p2 = np.array(interval1[0]), np.array(interval1[1])
    q1, q2 = np.array(interval2[0]), np.array(interval2[1])

    def point_seg_dist(p, a, b):
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
        t = max(0, min(1, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    # Check if segments intersect
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def intersect(a, b, c, d):
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    if intersect(p1, p2, q1, q2):
        return 0.0

    # Otherwise compute minimal distance among endpoint-to-segment distances
    dists = [
        point_seg_dist(p1, q1, q2),
        point_seg_dist(p2, q1, q2),
        point_seg_dist(q1, p1, p2),
        point_seg_dist(q2, p1, p2),
    ]
    return float(min(dists))


def get_dist(candidate, selected):
    def get_dist_to_interval_to_candidate(item):
        return get_dist_to_interval(candidate[1].endpoints, item[1].endpoints)

    return min(map(get_dist_to_interval_to_candidate, selected))


def filter_candidates(line_data, candidates, max_gap):
    i, line = line_data
    selected = [line_data]
    left = [candidate for candidate in candidates if candidate[0] != i]
    while True:
        new_selected = []
        new_left = []
        for candidate in left:
            if get_dist(candidate, selected) <= max_gap:
                new_selected.append(candidate)
            else:
                new_left.append(candidate)
        selected.extend(new_selected)
        left = new_left
        if not left or not new_selected:
            break
    return selected


def try_with_continues(item_0, candidates, max_gap):
    used = set()

    _, line = item_0

    candidates = filter_candidates(item_0, candidates, max_gap)
    # Merge candidates: take min and max projection along the line
    all_points: List[Tuple[float, float]] = []
    all_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for k, seg in candidates:
        used.add(k)
        all_points.extend(seg.endpoints)
        all_lines.append(seg.endpoints)

    # Project points onto direction vector
    vec = np.array((0, 0))
    for point_1, point_2 in all_lines:
        vec_point = np.array(point_1) - np.array(point_2)
        diff_1 = vec + vec_point
        diff_2 = vec - vec_point
        if np.linalg.norm(diff_1) > np.linalg.norm(diff_2):
            vec = diff_1
        else:
            vec = diff_2

    vec = vec / np.linalg.norm(vec)

    min_pt, max_pt, _ = best_parallel_segment(all_points, vec)
    # min_pt, max_pt, _ = best_parallel_segment(all_lines)

    # projections = [np.dot(np.array(pt) - np.array(p1), vec) for pt in all_points]
    # min_pt = tuple(np.array(p1) + vec * min(projections))
    # max_pt = tuple(np.array(p1) + vec * max(projections))

    # Construct merged line
    vec = np.array(max_pt) - np.array(min_pt)
    # Build merged Interval
    dominant_interval = max(candidates, key=lambda d: d[1].length)[1]
    merged_interval = Interval(
        endpoints=(tuple(map(int, min_pt)), tuple(map(int, max_pt))),
        angle=float(np.rad2deg(np.arctan2(vec[1], vec[0]))),
        length=float(_distance(min_pt, max_pt)),
        classification=dominant_interval.classification,
        thickness=dominant_interval.thickness,
        angle_label=line.angle_label,
        segments=[segment for _, segment in candidates],
    )
    return merged_interval, used


def merge_lines(
    intervals: List[Interval],
    tol=2.0,
    max_gap=5.0,
    scale=1000,
    max_gap_scale_coeff=0.005,
    coll_thr_scale_coeff=0.003,
    angle_tolerance_degrees=2,
    detect_dashed=False,
) -> List[Interval]:
    max_gap = max([max_gap, scale * max_gap_scale_coeff])
    """
    Merge `Interval` segments that are aligned/overlapping into longer lines.

    Args:
        lines: list of `Interval` objects.
        tol: tolerance for considering points collinear.
        max_gap: maximum distance allowed between two collinear segments to merge.

    Returns:
        list of `Interval`: merged lines.
    """

    merged = []
    used = set()
    list_of_lines = list(enumerate(intervals))
    left = dict(list_of_lines)
    intervals = sorted(intervals, key=lambda x: -x.length)

    for i, line in tqdm(list_of_lines):
        if i in used:
            continue

        # p1, p2 = line.endpoints
        candidates = [(i, line)]

        used.add(i)

        # Look for other lines that connect or overlap
        for j, other in left.items():
            if i == j : continue
            # Check if collinear and not too far apart
            if _lines_has_same_angle(
                line,
                other,
                scale=scale,
                angle_tolerance_degrees=angle_tolerance_degrees,
                coll_thr_scale_coeff=coll_thr_scale_coeff,
            ):
                candidates.append((j, other))

        if len(candidates) == 1:
            merged.append(line)
            used.add(i)
            left.pop(i)
            continue

        # new_interval, used_ids = try_with_dasshes(
        #     (i, line), candidates, intervals
        # ) or ( None, {}) # or try_with_continues((i, line), candidates, max_gap)
        # # print(used_ids, left)

        new_interval, used_ids = (try_with_dasshes(
            (i, line), candidates, intervals, scale=scale,
        ) if detect_dashed else None) or try_with_continues((i, line), candidates, max_gap)
        

        
        for used_id in used_ids:
            used.add(used_id)
            left.pop(used_id)
        if new_interval:
            merged.append(new_interval)
    return merged
    # for i, line in tqdm(list_of_lines):
    #     if i in used:
    #         continue

    #     # p1, p2 = line.endpoints
    #     candidates = [(i, line)]

    #     used.add(i)

    #     # Look for other lines that connect or overlap
    #     for j, other in left.items():
    #         # Check if collinear and not too far apart
    #         if _lines_has_same_angle(
    #             line,
    #             other,
    #             scale=scale,
    #             angle_tolerance_degrees=angle_tolerance_degrees,
    #             coll_thr_scale_coeff=coll_thr_scale_coeff,
    #         ):
    #             candidates.append((j, other))

    #     if len(candidates) == 1:
    #         merged.append(line)
    #         used.add(i)
    #         left.pop(i)
    #         continue

    #     new_interval, used_ids = try_with_continues((i, line), candidates, max_gap)
    #     # try_with_dasshes(
    #     #     (i, line), candidates, intervals
    #     # ) or 

    #     for used_id in used_ids:
    #         used.add(used_id)
    #         left.pop(used_id)

    #     merged.append(new_interval)

    # return merged

def merge_lines_v1(
    intervals: List[Interval],
    tol=2.0,
    max_gap=5.0,
    scale=1000,
    max_gap_scale_coeff=0.005,
    coll_thr_scale_coeff=0.003,
    angle_tolerance_degrees=2,
) -> List[Interval]:
    max_gap = max([max_gap, scale * max_gap_scale_coeff])
    """
    Merge `Interval` segments that are aligned/overlapping into longer lines.

    Args:
        lines: list of `Interval` objects.
        tol: tolerance for considering points collinear.
        max_gap: maximum distance allowed between two collinear segments to merge.

    Returns:
        list of `Interval`: merged lines.
    """

    merged = []
    used = set()
    list_of_lines = list(enumerate(intervals))
    left = dict(list_of_lines)
    intervals = sorted(intervals, key=lambda x: -x.length)

    for i, line in tqdm(list_of_lines):
        if i in used:
            continue

        # p1, p2 = line.endpoints
        candidates = [(i, line)]

        used.add(i)

        # Look for other lines that connect or overlap
        for j, other in left.items():
            if i ==j: continue
            # Check if collinear and not too far apart
            if _lines_has_same_angle(
                line,
                other,
                scale=scale,
                angle_tolerance_degrees=angle_tolerance_degrees,
                coll_thr_scale_coeff=coll_thr_scale_coeff,
            ):
                candidates.append((j, other))

        if len(candidates) == 1:
            merged.append(line)
            used.add(i)
            left.pop(i)
            continue

        new_interval, used_ids = try_with_dasshes(
            (i, line), candidates, intervals
        ) or try_with_continues((i, line), candidates, max_gap)

        for used_id in used_ids:
            used.add(used_id)
            left.pop(used_id)
        if new_interval:
            merged.append(new_interval)


    return merged


# Example usage
# lines = [
#     {"endpoints": ((0, 0), (1, 1)), "angle": 45, "length": 1.41, "type": "thin"},
#     {"endpoints": ((1, 1), (2, 2)), "angle": 45, "length": 1.41, "type": "thick"},
#     {"endpoints": ((4, 4), (5, 5)), "angle": 45, "length": 1.41, "type": "thick"},
# ]

# merged = merge_lines(lines)
# for m in merged:
#     print(m)

# merge_lines(found, scale=3500)


def group_lines_into_dashed(lines):
    """Improved method to find and group segments that form a dashed line."""
    if len(lines) < DASHED_GROUP_MIN_LINES:
        return [], lines

    line_indices = {id(line): i for i, line in enumerate(lines)}
    grouped_flags = [False] * len(lines)
    dashed_lines = []

    for i in range(len(lines)):
        if grouped_flags[i]:
            continue

        seed_line = lines[i]

        # 1. Form a potential group of collinear lines with similar length
        group = [seed_line]
        avg_len = seed_line["length"]
        for j in range(i + 1, len(lines)):
            if grouped_flags[j]:
                continue
            candidate = lines[j]

            angle_diff = abs(seed_line["angle"] - candidate["angle"])
            if min(angle_diff, abs(angle_diff - 180)) > MERGE_ANGLE_TOLERANCE_DEGREES:
                continue
            if abs(candidate["length"] - avg_len) / avg_len > DASHED_LENGTH_TOLERANCE:
                continue

            group.append(candidate)

        if len(group) < DASHED_GROUP_MIN_LINES:
            continue

        # 2. Sort the group members along their common axis for proper gap analysis
        ref_point = np.array(group[0]["endpoints"][0])
        ref_vec = np.array(group[0]["endpoints"][1]) - ref_point
        group.sort(
            key=lambda l: np.dot(np.array(l["endpoints"][0]) - ref_point, ref_vec)
        )

        # 3. Analyze the gaps between the sorted segments
        gaps = []
        for k in range(len(group) - 1):
            line_a, line_b = group[k], group[k + 1]
            min_dist = min(
                point_distance(p_a, p_b)
                for p_a in line_a["endpoints"]
                for p_b in line_b["endpoints"]
            )
            gaps.append(min_dist)

        if not gaps:
            continue

        # 4. Check if gaps are regular and not too large
        avg_gap = sum(gaps) / len(gaps)
        avg_len = sum(l["length"] for l in group) / len(group)
        if avg_gap == 0 or avg_gap > avg_len * DASHED_MAX_GAP_FACTOR:
            continue

        is_regular = all(
            abs(g - avg_gap) / avg_gap < DASHED_GAP_TOLERANCE for g in gaps
        )

        # 5. If it's a valid dashed line, create the final representation
        if is_regular:
            for line_in_group in group:
                grouped_flags[line_indices[id(line_in_group)]] = True

            all_points = [p for l in group for p in l["endpoints"]]
            max_dist = -1
            final_p1, final_p2 = None, None
            for p_start_idx in range(len(all_points)):
                for p_end_idx in range(p_start_idx + 1, len(all_points)):
                    dist = point_distance(
                        all_points[p_start_idx], all_points[p_end_idx]
                    )
                    if dist > max_dist:
                        max_dist = dist
                        final_p1 = all_points[p_start_idx]
                        final_p2 = all_points[p_end_idx]
            dashed_lines.append(
                {
                    "points": (final_p1[0], final_p1[1], final_p2[0], final_p2[1]),
                    "type": "dashed",
                }
            )

    remaining_lines = [lines[i] for i in range(len(lines)) if not grouped_flags[i]]
    return dashed_lines, remaining_lines


def try_with_dasshes(item_0, candidates, all_intrevals,
    scale=1000,
    min_segments_required = 3,
    length_rel_tolerance = 1.3,  # allow ~30% variation in segment length
    gap_rel_tolerance = 1.35  # allow ~35% variation in gaps
):
    # Robust dashed-line detection around a base segment by expanding left/right
    # and dropping outliers (large gaps or odd lengths).

    min_segments_required = 3
    length_rel_tolerance = 0.3  # allow ~30% variation in segment length
    gap_rel_tolerance = 0.35  # allow ~35% variation in gaps

    # Unpack
    i0, seed = item_0
    if seed.length > 0.01 * scale:
        return None
    if len(candidates) < min_segments_required:
        return None

    # Build reference direction from seed
    (sx1, sy1), (sx2, sy2) = seed.endpoints
    ref_vec = np.array([sx2 - sx1, sy2 - sy1], dtype=float)
    if np.linalg.norm(ref_vec) < 1e-9:
        return None
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    origin = np.array([sx1, sy1], dtype=float)

    def proj_scalar(pt):
        return float(np.dot(np.array(pt, dtype=float) - origin, ref_vec))

    # Sort by projection of midpoints
    sortable = []  # (proj, id, interval)
    for cid, seg in candidates:
        # print(cid)
        (x1, y1), (x2, y2) = seg.endpoints
        mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        sortable.append((proj_scalar(mid), cid, seg))
    sortable.sort(key=lambda t: t[0])

    # Find base index in sorted list
    base_idx = None
    for idx, (_, cid, _) in enumerate(sortable):
        if cid == i0:
            base_idx = idx
            break
    if base_idx is None:
        return None

    
    # Helpers
    def min_endpoint_gap(a: Interval, b: Interval) -> float:
        ax1, ay1 = a.endpoints[0]
        ax2, ay2 = a.endpoints[1]
        bx1, by1 = b.endpoints[0]
        bx2, by2 = b.endpoints[1]
        pts_a = [(ax1, ay1), (ax2, ay2)]
        pts_b = [(bx1, by1), (bx2, by2)]
        return min(_distance(pa, pb) for pa in pts_a for pb in pts_b)

    accepted_indices = {base_idx}
    current_lengths = [sortable[base_idx][2].length]
    current_gaps = []

    # Expand to the right
    k = base_idx
    while k < len(sortable) - 1:
        _, _, left_seg = sortable[k]
        _, cand_id, right_seg = sortable[k + 1]
        gap = min_endpoint_gap(left_seg, right_seg)
        mean_len = float(np.mean(current_lengths))
        mean_gap = float(np.mean(current_gaps)) if current_gaps else gap
        if mean_len <= 0:
            break
        length_ok = abs(right_seg.length - mean_len) / mean_len <= length_rel_tolerance
        gap_ok = (
            abs(gap - mean_gap) / (mean_gap if mean_gap > 0 else 1.0)
            <= gap_rel_tolerance
        )
        if length_ok and gap_ok:
            accepted_indices.add(k + 1)
            current_lengths.append(right_seg.length)
            current_gaps.append(gap)
            k += 1
        else:
            break

    # Expand to the left
    k = base_idx
    while k > 0:
        _, _, right_seg = sortable[k]
        _, cand_id, left_seg = sortable[k - 1]
        gap = min_endpoint_gap(left_seg, right_seg)
        mean_len = float(np.mean(current_lengths))
        mean_gap = float(np.mean(current_gaps)) if current_gaps else gap
        if mean_len <= 0:
            break
        length_ok = abs(left_seg.length - mean_len) / mean_len <= length_rel_tolerance
        gap_ok = (
            abs(gap - mean_gap) / (mean_gap if mean_gap > 0 else 1.0)
            <= gap_rel_tolerance
        )
        if length_ok and gap_ok:
            accepted_indices.add(k - 1)
            current_lengths.append(left_seg.length)
            current_gaps.append(gap)
            k -= 1
        else:
            break

    # Require minimum number of segments
    if len(accepted_indices) < min_segments_required:
        return None

    

    # Build merged dashed interval over accepted subset only
    accepted = [sortable[idx] for idx in sorted(accepted_indices)]
    used_ids = set(cid for _, cid, _ in accepted)
    print(accepted_indices)
    # return [
    #     t[2] for t in accepted
    # ], [cid for _, cid, _ in accepted]


    all_points = []
    for _, cid, seg in accepted:
        all_points.extend(seg.endpoints)

    min_pt, max_pt, _ = best_parallel_segment(all_points, tuple(ref_vec))
    span_vec = np.array(max_pt) - np.array(min_pt)
    dominant_interval = max((seg for _, _, seg in accepted), key=lambda s: s.length)

    merged_interval = Interval(
        endpoints=(tuple(map(int, min_pt)), tuple(map(int, max_pt))),
        angle=float(np.rad2deg(np.arctan2(span_vec[1], span_vec[0]))),
        length=float(_distance(min_pt, max_pt)),
        classification="dashed",
        thickness=dominant_interval.thickness,
        angle_label=seed.angle_label,
        segments=[seg for _, _, seg in accepted],
    )
    print(merged_interval)
    return merged_interval, used_ids
