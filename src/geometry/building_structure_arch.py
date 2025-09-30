from typing import Dict, List, Optional, Set, Tuple
from itertools import product, chain
from collections import defaultdict

import cv2
import numpy as np
from pydantic import BaseModel, Field

from src.geometry.objects import Interval
from tqdm import tqdm

class Shape(BaseModel):
    """
    Aggregation of intervals forming a coherent geometric entity.

    - If is_closed is True, polygon holds ordered vertices approximating a closed shape.
    - If open, polygon may be None and intervals describe partial structure (e.g., single wall).
    """

    kind: str = Field(
        default="unknown", description="Semantic label (e.g., 'room', 'table', 'support')."
    )
    intervals: List[Interval] = Field(default_factory=list)
    polygon: Optional[List[Tuple[float, float]]] = None
    is_closed: bool = False
    assisting_intervals: List[Interval] = Field(
        default_factory=list,
        description="Intervals considered assisting (e.g., scale lines) to render in red.",
    )


def _snap_points(points: List[Tuple[float, float]], tol: float) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Cluster points within tolerance and return canonical points and mapping indices.
    Returns (centers, mapping) where mapping[i] = idx into centers for points[i].
    """
    centers: List[Tuple[float, float]] = []
    mapping: List[int] = []
    for px, py in points:
        if not centers:
            centers.append((px, py))
            mapping.append(0)
            continue
        found = False
        for ci, (cx, cy) in enumerate(centers):
            if np.hypot(px - cx, py - cy) <= tol:
                # simple running-average update to stabilize center
                nx = (cx + px) / 2.0
                ny = (cy + py) / 2.0
                centers[ci] = (nx, ny)
                mapping.append(ci)
                found = True
                break
        if not found:
            centers.append((px, py))
            mapping.append(len(centers) - 1)
    return centers, mapping


def _build_graph(intervals: List[Interval], tol: float) -> Tuple[Dict[int, Set[int]], List[Tuple[int, int]], List[Tuple[float, float]]]:
    """
    Build a graph of snapped endpoints.
    Returns adjacency, edge list (u,v), and centers list (node -> point).
    """
    endpoints: List[Tuple[float, float]] = []
    for seg in intervals:
        endpoints.extend([tuple(seg.endpoints[0]), tuple(seg.endpoints[1])])
    centers, mapping = _snap_points(endpoints, tol)
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(len(centers))}
    edges: List[Tuple[int, int]] = []
    for idx, seg in enumerate(intervals):
        a_idx = mapping[2 * idx]
        b_idx = mapping[2 * idx + 1]
        if a_idx == b_idx:
            continue
        adjacency[a_idx].add(b_idx)
        adjacency[b_idx].add(a_idx)
        edges.append((a_idx, b_idx))
    return adjacency, edges, centers


def _map_interval_to_existing_nodes(
    centers: List[Tuple[float, float]],
    interval: Interval,
    tol: float,
) -> Optional[Tuple[int, int]]:
    """
    If both endpoints of interval are within tol from some existing centers, return their indices.
    """
    a_pt = tuple(interval.endpoints[0])
    b_pt = tuple(interval.endpoints[1])
    ai = None
    bi = None
    for idx, c in enumerate(centers):
        if ai is None and np.hypot(a_pt[0] - c[0], a_pt[1] - c[1]) <= tol:
            ai = idx
        if bi is None and np.hypot(b_pt[0] - c[0], b_pt[1] - c[1]) <= tol:
            bi = idx
        if ai is not None and bi is not None:
            break
    if ai is None or bi is None or ai == bi:
        return None
    return (ai, bi)


def _map_interval_to_existing_nodes_with_margin(
    centers: List[Tuple[float, float]],
    interval: Interval,
    tol: float,
    margin_factor: float = 2.5,
) -> Optional[Tuple[int, int]]:
    """
    Like _map_interval_to_existing_nodes but allows a larger tolerance to catch
    indirect connections with some margin of error.
    """
    margin = tol * margin_factor
    a_pt = tuple(interval.endpoints[0])
    b_pt = tuple(interval.endpoints[1])
    ai = None
    bi = None
    best_da = 1e18
    best_db = 1e18
    for idx, c in enumerate(centers):
        da = np.hypot(a_pt[0] - c[0], a_pt[1] - c[1])
        db = np.hypot(b_pt[0] - c[0], b_pt[1] - c[1])
        if da <= margin and da < best_da:
            ai = idx
            best_da = da
        if db <= margin and db < best_db:
            bi = idx
            best_db = db
    if ai is None or bi is None or ai == bi:
        return None
    return (ai, bi)


def _project_point_to_segment(
    point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[Tuple[float, float], float, float]:
    """
    Project point onto segment ab. Returns (proj_point, t, distance), where
    proj = a + t*(b-a), t in R (not clamped), distance is Euclidean to proj.
    """
    p = np.array(point, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return (tuple(a), 0.0, float(np.linalg.norm(p - a)))
    t = float(np.dot(p - a, ab) / denom)
    proj = a + t * ab
    return (tuple(proj.tolist()), t, float(np.linalg.norm(p - proj)))


def _split_intervals_at_tjunctions(
    intervals: List[Interval],
    tol_dist: float,
    tol_t_margin: float = 1e-3,
) -> List[Interval]:
    """
    Split intervals when an endpoint of any interval lands (within tol_dist) on the interior
    of another interval. Keeps geometry close to original; produces sub-intervals.
    """
    result: List[Interval] = []
    for idx_b, base in enumerate(intervals):
        split_params: List[float] = []
        a_b = tuple(base.endpoints[0])
        b_b = tuple(base.endpoints[1])
        for idx_o, other in enumerate(intervals):
            if idx_o == idx_b:
                continue
            # Test both endpoints of 'other' against segment 'base'
            for ep in [tuple(other.endpoints[0]), tuple(other.endpoints[1])]:
                proj, t, d = _project_point_to_segment(ep, a_b, b_b)
                if d <= tol_dist and tol_t_margin < t < 1.0 - tol_t_margin:
                    split_params.append(t)
        if not split_params:
            result.append(base)
            continue
        # Create sorted unique split parameters
        splits = sorted(set(split_params))
        # Build sub-intervals along base
        a = np.array(a_b, dtype=float)
        b = np.array(b_b, dtype=float)
        points = [a] + [a + t * (b - a) for t in splits] + [b]
        for p1, p2 in zip(points[:-1], points[1:]):
            if float(np.linalg.norm(p2 - p1)) < 1e-6:
                continue
            vec = p2 - p1
            angle = float(np.rad2deg(np.arctan2(vec[1], vec[0])))
            length = float(np.linalg.norm(vec))
            result.append(
                Interval(
                    endpoints=(tuple(map(float, p1)), tuple(map(float, p2))),
                    angle=angle,
                    length=length,
                    classification=base.classification,
                    thickness=base.thickness,
                    angle_label=base.angle_label,
                    segments=base.segments or [base],
                )
            )
    return result


def _find_cycles_limited(adjacency: Dict[int, Set[int]], max_len: int = 8) -> List[List[int]]:
    """
    Enumerate simple cycles up to length max_len using bounded DFS from each node.
    Deduplicate cycles by normalized representation (rotate to smallest node id, enforce direction).
    """
    cycles: Set[Tuple[int, ...]] = set()
    visited_stack: List[int] = []

    def dfs(start: int, current: int, parent: int, depth: int):
        if depth > max_len:
            return
        visited_stack.append(current)
        for nxt in adjacency[current]:
            if nxt == parent:
                continue
            if nxt == start and depth >= 3:
                cyc = visited_stack.copy()
                # close cycle
                cyc_tuple = tuple(cyc)
                # normalize by rotation and direction
                mins = min(cyc_tuple)
                # rotate
                idx = cyc_tuple.index(mins)
                rot1 = cyc_tuple[idx:] + cyc_tuple[:idx]
                rot2 = tuple(reversed(rot1))
                norm = tuple(rot1) if rot1 < rot2 else rot2
                cycles.add(norm)
                continue
            if nxt in visited_stack:
                continue
            dfs(start, nxt, current, depth + 1)
        visited_stack.pop()

    for node in adjacency.keys():
        dfs(node, node, -1, 1)
    return [list(c) for c in cycles]


def _edge_exists(edges: List[Tuple[int, int]], u: int, v: int) -> bool:
    return (u, v) in edges or (v, u) in edges


def _segments_intersect(a1: Tuple[float, float], a2: Tuple[float, float], b1: Tuple[float, float], b2: Tuple[float, float]) -> bool:
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))


def _angle_between(seg_a: Interval, seg_b: Interval) -> float:
    ax1, ay1 = seg_a.endpoints[0]
    ax2, ay2 = seg_a.endpoints[1]
    bx1, by1 = seg_b.endpoints[0]
    bx2, by2 = seg_b.endpoints[1]
    va = np.array([ax2 - ax1, ay2 - ay1], dtype=float)
    vb = np.array([bx2 - bx1, by2 - by1], dtype=float)
    if np.linalg.norm(va) < 1e-9 or np.linalg.norm(vb) < 1e-9:
        return 0.0
    va = va / np.linalg.norm(va)
    vb = vb / np.linalg.norm(vb)
    dot = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(abs(dot))))
    return angle


def _interval_from_nodes(
    centers: List[Tuple[float, float]],
    a: int,
    b: int,
    template: Optional[Interval] = None,
) -> Interval:
    p1 = centers[a]
    p2 = centers[b]
    vec = np.array(p2) - np.array(p1)
    angle = float(np.rad2deg(np.arctan2(vec[1], vec[0])))
    length = float(np.hypot(vec[0], vec[1]))
    return Interval(
        endpoints=(tuple(map(float, p1)), tuple(map(float, p2))),
        angle=angle,
        length=length,
        classification=(template.classification if template else None),
        thickness=(template.thickness if template else None),
        angle_label=(template.angle_label if template else None),
    )


def connect_main_lines(
    main_intervals: List[Interval],
    all_intervals: List[Interval],
    img_binary: Optional[np.ndarray],
    snap_tol: Optional[float] = None,
) -> List[Shape]:
    """
    Aggregate main_intervals into clusters forming closed or open shapes.

    Strategy:
    - Snap endpoints into graph nodes using a tolerance based on image size (or default).
    - Build a graph of main edges; augment with candidate edges from all_intervals and image ink.
    - Detect small cycles to form closed shapes; create open shapes for leftover components.
    """
    if not main_intervals:
        return []

    # Determine snapping tolerance
    if snap_tol is None:
        if img_binary is not None and img_binary.size > 0:
            h, w = img_binary.shape[:2]
            snap_tol = max(3.0, 0.003 * float(np.hypot(h, w)))
        else:
            # fallback without image context
            avg_len = float(np.mean([seg.length for seg in main_intervals])) if main_intervals else 10.0
            snap_tol = max(3.0, 0.02 * avg_len)

    # Pre-split intervals at T-junctions to enable branch alternatives
    split_tol = max(2.0, 0.5 * snap_tol)
    refined_main = _split_intervals_at_tjunctions(main_intervals, tol_dist=split_tol)
    refined_all = _split_intervals_at_tjunctions(all_intervals or [], tol_dist=split_tol) if all_intervals else []

    adjacency, edges, centers = _build_graph(refined_main, tol=snap_tol)

    # Augment edges using all_intervals as connectors between existing snapped nodes
    if refined_all:
        for inter in refined_all:
            mapped = _map_interval_to_existing_nodes(centers, inter, tol=snap_tol)
            if mapped is None:
                mapped = _map_interval_to_existing_nodes_with_margin(centers, inter, tol=snap_tol)
            if mapped is None:
                continue
            a, b = mapped
            if not _edge_exists(edges, a, b):
                edges.append((a, b))
                adjacency[a].add(b)
                adjacency[b].add(a)
    # Graph ready; no image-based gap filling to keep logic simple

    print(f"adjacency {adjacency}")
    print(f"edges {edges}")
    print(f"centers {centers}")

    # Find cycles to create closed shapes
    cycles = _find_cycles_limited(adjacency, max_len=8)
    used_edges: Set[Tuple[int, int]] = set()
    room_shapes_raw: List[Shape] = []
    for cyc in cycles:
        edges_in_cyc: List[Tuple[int, int]] = []
        ok = True
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            if not _edge_exists(edges, u, v):
                ok = False
                break
            edges_in_cyc.append((min(u, v), max(u, v)))
        if not ok:
            continue
        intervals_for_shape: List[Interval] = []
        for u, v in edges_in_cyc:
            used_edges.add((u, v))
            cu, cv = centers[u], centers[v]
            matched: Optional[Interval] = None
            for seg in list(refined_main) + list(refined_all or []):
                e1, e2 = tuple(seg.endpoints[0]), tuple(seg.endpoints[1])
                if max(np.hypot(cu[0] - e1[0], cu[1] - e1[1]), np.hypot(cv[0] - e2[0], cv[1] - e2[1])) <= snap_tol or \
                   max(np.hypot(cu[0] - e2[0], cu[1] - e2[1]), np.hypot(cv[0] - e1[0], cv[1] - e1[1])) <= snap_tol:
                    matched = seg
                    break
            if matched is None:
                matched = _interval_from_nodes(centers, u, v)
            intervals_for_shape.append(matched)
        if intervals_for_shape:
            corrected_intervals = connect_cycle(intervals_for_shape, snap_tol * 3)
            if corrected_intervals is None:
                continue
            room_shapes_raw.append(
                Shape(kind="room", intervals=corrected_intervals, polygon=None, is_closed=True)
            )

    # Select closest cycles (left/right) per main interval using _id-based de-dup
    selected_keys: Set[Tuple[str, ...]] = set()
    shapes: List[Shape] = []
    for seed in main_intervals:
        picked = find_closest_cycles(seed, room_shapes_raw)
        for shp in picked:
            if not shp.polygon or len(shp.polygon) < 3:
                shp.polygon = _cycle_vertices(shp.intervals)
            key = tuple(sorted(_id(iv) for iv in shp.intervals))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            shapes.append(shp)

    # Remaining intervals (not fully used in closed shapes) become open shapes by connected components
    # Build a masked graph excluding edges used in closed shapes
    n_nodes = len(centers)
    masked_adj: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
    used_edge_set = set(used_edges)
    for u, neigh in adjacency.items():
        for v in neigh:
            a, b = (min(u, v), max(u, v))
            if (a, b) in used_edge_set:
                continue
            masked_adj[u].add(v)

    visited: Set[int] = set()
    for start in range(n_nodes):
        if start in visited:
            continue
        # BFS to collect a component
        queue = [start]
        comp_nodes: List[int] = []
        visited.add(start)
        while queue:
            x = queue.pop(0)
            comp_nodes.append(x)
            for y in masked_adj[x]:
                if y not in visited:
                    visited.add(y)
                    queue.append(y)
        # Gather intervals connecting nodes in this component
        comp_edges: Set[Tuple[int, int]] = set()
        for u in comp_nodes:
            for v in masked_adj[u]:
                if u < v:
                    comp_edges.add((u, v))
        if not comp_edges:
            continue
        intervals_for_open: List[Interval] = []
        for u, v in comp_edges:
            matched: Optional[Interval] = None
            cu, cv = centers[u], centers[v]
            for seg in list(refined_main) + list(refined_all or []):
                e1, e2 = tuple(seg.endpoints[0]), tuple(seg.endpoints[1])
                if max(np.hypot(cu[0] - e1[0], cu[1] - e1[1]), np.hypot(cv[0] - e2[0], cv[1] - e2[1])) <= snap_tol or \
                   max(np.hypot(cu[0] - e2[0], cu[1] - e2[1]), np.hypot(cv[0] - e1[0], cv[1] - e1[1])) <= snap_tol:
                    matched = seg
                    break
            if matched is None:
                matched = _interval_from_nodes(centers, u, v)
                intervals_for_open.append(matched)
            else:
                shared = 0
                for c in centers:
                    if min(np.hypot(matched.endpoints[0][0]-c[0], matched.endpoints[0][1]-c[1]), np.hypot(matched.endpoints[1][0]-c[0], matched.endpoints[1][1]-c[1])) <= snap_tol:
                        shared += 1
                crosses = 0
                for other in list(refined_main) + list(refined_all or []):
                    if other is matched:
                        continue
                    if _segments_intersect(matched.endpoints[0], matched.endpoints[1], other.endpoints[0], other.endpoints[1]):
                        if _angle_between(matched, other) > 10.0:
                            crosses += 1
                            break
                if crosses >= 1 and shared <= 1:
                    # classify as assisting, but keep structural synthetic interval
                    synthetic = _interval_from_nodes(centers, u, v, template=matched)
                    intervals_for_open.append(synthetic)
                    # store assisting after shape creation
                    intervals_for_open.append(synthetic)
                else:
                    intervals_for_open.append(matched)
        shp = Shape(kind="support", intervals=intervals_for_open, polygon=None, is_closed=False)
        # Attach assisting that were used in place of edges
        for u, v in comp_edges:
            cu, cv = centers[u], centers[v]
            for seg in list(refined_main) + list(refined_all or []):
                e1, e2 = tuple(seg.endpoints[0]), tuple(seg.endpoints[1])
                if max(np.hypot(cu[0] - e1[0], cu[1] - e1[1]), np.hypot(cv[0] - e2[0], cv[1] - e2[1])) <= snap_tol or \
                   max(np.hypot(cu[0] - e2[0], cu[1] - e2[1]), np.hypot(cv[0] - e1[0], cv[1] - e1[1])) <= snap_tol:
                    shared = 0
                    for c in centers:
                        if min(np.hypot(seg.endpoints[0][0]-c[0], seg.endpoints[0][1]-c[1]), np.hypot(seg.endpoints[1][0]-c[0], seg.endpoints[1][1]-c[1])) <= snap_tol:
                            shared += 1
                    crosses = 0
                    for other in list(refined_main) + list(refined_all or []):
                        if other is seg:
                            continue
                        if _segments_intersect(seg.endpoints[0], seg.endpoints[1], other.endpoints[0], other.endpoints[1]):
                            if _angle_between(seg, other) > 10.0:
                                crosses += 1
                                break
                    if crosses >= 1 and shared <= 1:
                        if seg not in shp.assisting_intervals:
                            shp.assisting_intervals.append(seg)
        shapes.append(shp)

    # Assisting lines are now only added per shape when a candidate edge behaved as assisting.

    return shapes


def compute_interval_distance(interval_a: Interval, interval_b: Interval, snap_tol: float) -> float:
    """
    Distance between two intervals; returns 0 if they intersect or overlap within tolerance.
    """
    a1, a2 = tuple(interval_a.endpoints[0]), tuple(interval_a.endpoints[1])
    b1, b2 = tuple(interval_b.endpoints[0]), tuple(interval_b.endpoints[1])
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0
    # endpoint-to-segment distances
    def point_seg_dist(p, a, b):
        ap = np.array(p) - np.array(a)
        ab = np.array(b) - np.array(a)
        t = float(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12))
        t = max(0.0, min(1.0, t))
        proj = np.array(a) + t * ab
        return float(np.linalg.norm(np.array(p) - proj))
    dists = [
        point_seg_dist(a1, b1, b2),
        point_seg_dist(a2, b1, b2),
        point_seg_dist(b1, a1, a2),
        point_seg_dist(b2, a1, a2),
    ]
    return float(min(dists))


def _build_local_graph(intervals: List[Interval], snap_tol: float):
    """Build snapped graph from given intervals; returns adjacency, edges, centers."""
    return _build_graph(intervals, tol=snap_tol)


def _dedup_shapes(shapes: List[Shape], tol: float) -> List[Shape]:
    """Deduplicate shapes by polygon geometry or interval signatures."""
    seen: Set[Tuple] = set()
    out: List[Shape] = []
    for s in shapes:
        if s.is_closed and s.polygon:
            key = tuple((int(round(x / tol)), int(round(y / tol))) for (x, y) in s.polygon)
        else:
            sig = []
            for seg in s.intervals:
                p = tuple(int(round(v / tol)) for v in seg.endpoints[0])
                q = tuple(int(round(v / tol)) for v in seg.endpoints[1])
                sig.append(tuple(sorted([p, q])))
            key = tuple(sorted(sig))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def shapes_for_interval(
    seed: Interval,
    prox_main: Dict[int, List[Interval]],
    prox_alt: Dict[int, List[Interval]],
    snap_tol: float,
) -> List[Shape]:
    """
    From a seed interval, build a local set of nearby intervals (main + connectors) and
    detect small cycles to form shapes. Assisting classification happens when selecting edges.
    """
    local_intervals: List[Interval] = [seed] + prox_main.get(id(seed), []) + prox_alt.get(id(seed), [])
    if len(local_intervals) <= 1:
        return []
    adjacency, edges, centers = _build_local_graph(local_intervals, snap_tol)
    cycles = _find_cycles_limited(adjacency, max_len=8)
    shapes: List[Shape] = []
    # Build shapes for cycles
    for cyc in cycles:
        polygon = [centers[i] for i in cyc]
        intervals_for_shape: List[Interval] = []
        assisting_for_shape: List[Interval] = []
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            cu, cv = centers[u], centers[v]
            matched: Optional[Interval] = None
            for seg in local_intervals:
                e1, e2 = tuple(seg.endpoints[0]), tuple(seg.endpoints[1])
                if max(np.hypot(cu[0]-e1[0], cu[1]-e1[1]), np.hypot(cv[0]-e2[0], cv[1]-e2[1])) <= snap_tol or \
                   max(np.hypot(cu[0]-e2[0], cu[1]-e2[1]), np.hypot(cv[0]-e1[0], cv[1]-e1[1])) <= snap_tol:
                    matched = seg
                    break
            if matched is None:
                matched = _interval_from_nodes(centers, u, v)
                intervals_for_shape.append(matched)
            else:
                # Assist classification only when chosen as an edge
                shared = 0
                for c in centers:
                    if min(np.hypot(matched.endpoints[0][0]-c[0], matched.endpoints[0][1]-c[1]), np.hypot(matched.endpoints[1][0]-c[0], matched.endpoints[1][1]-c[1])) <= snap_tol:
                        shared += 1
                crosses = 0
                for other in local_intervals:
                    if other is matched:
                        continue
                    if _segments_intersect(matched.endpoints[0], matched.endpoints[1], other.endpoints[0], other.endpoints[1]) and _angle_between(matched, other) > 10.0:
                        crosses += 1
                        break
                if crosses >= 1 and shared <= 1:
                    synthetic = _interval_from_nodes(centers, u, v, template=matched)
                    intervals_for_shape.append(synthetic)
                    assisting_for_shape.append(matched)
                else:
                    intervals_for_shape.append(matched)
        if intervals_for_shape:
            shapes.append(Shape(kind="room", intervals=intervals_for_shape, polygon=polygon, is_closed=True, assisting_intervals=assisting_for_shape))
    # If no cycles, return an open support shape made of local mains
    if not shapes:
        shapes.append(Shape(kind="support", intervals=list({id(i): i for i in local_intervals}.values()), polygon=None, is_closed=False))
    return shapes


def connect_main_lines_apporach_b(
    main_intervals: List[Interval],
    all_intervals: List[Interval],
    img_binary: Optional[np.ndarray],
    snap_tol: Optional[float] = None,
) -> List[Shape]:
    # Configure tolerance
    if not main_intervals:
        return []
    if snap_tol is None:
        avg_len = float(np.mean([seg.length for seg in main_intervals])) if main_intervals else 10.0
        snap_tol = max(3.0, 0.02 * avg_len)
    all_intervals_mod = [item for item in (all_intervals or []) if item not in main_intervals]
    # Build proximity maps
    proximity: Dict[int, List[Interval]] = defaultdict(list)
    for a, b in product(main_intervals, main_intervals):
        if a is b:
            continue
        if compute_interval_distance(a, b, snap_tol) <= snap_tol:
            proximity[id(a)].append(b)
    proximity_alt: Dict[int, List[Interval]] = defaultdict(list)
    for a, b in product(main_intervals, all_intervals_mod):
        if compute_interval_distance(a, b, snap_tol) <= snap_tol:
            proximity_alt[id(a)].append(b)
    # Build proximity among rest (connectors graph)
    rest_ids: Dict[int, Interval] = {id(r): r for r in all_intervals_mod}
    rest_adj: Dict[int, List[int]] = defaultdict(list)
    rest_list = list(rest_ids.items())
    for (id1, r1), (id2, r2) in tqdm(product(rest_list, rest_list), total=len(rest_list)**2):
        if id1 == id2:
            continue
        if compute_interval_distance(r1, r2, snap_tol) <= snap_tol:
            rest_adj[id1].append(id2)
    
    # Helper to find a connector chain of length <= max_hops between two mains via rest
    def find_connector_chain(seed_a: Interval, seed_b: Interval, max_hops: int = 3) -> List[Interval]:
        start_neighbors = [id(r) for r in proximity_alt.get(id(seed_a), [])]
        target_set = set([id(r) for r in proximity_alt.get(id(seed_b), [])])
        if not start_neighbors:
            return []
        if target_set:
            # direct one-hop via same rest element
            for rid in start_neighbors:
                if rid in target_set:
                    return [rest_ids[rid]]
        # BFS up to max_hops in rest graph
        from collections import deque
        queue = deque()
        visited: Set[int] = set()
        parent: Dict[int, Optional[int]] = {}
        for rid in start_neighbors:
            queue.append((rid, 1))
            visited.add(rid)
            parent[rid] = None
        while queue:
            cur, depth = queue.popleft()
            if cur in target_set:
                # reconstruct path
                path_ids: List[int] = []
                x = cur
                while x is not None:
                    path_ids.append(x)
                    x = parent[x]
                path_ids.reverse()
                return [rest_ids[pid] for pid in path_ids]
            if depth >= max_hops:
                continue
            for nxt in rest_adj.get(cur, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                parent[nxt] = cur
                queue.append((nxt, depth + 1))
        return []

    # Augment per-seed connectors with rest chains for distant main pairs
    bridge_alt: Dict[int, List[Interval]] = defaultdict(list)
    for a, b in tqdm(product(main_intervals, main_intervals)):
        if a is b:
            continue
        if compute_interval_distance(a, b, snap_tol) <= snap_tol:
            continue
        # First check if there is any single rest close to both
        set_a = set(id(r) for r in proximity_alt.get(id(a), []))
        set_b = set(id(r) for r in proximity_alt.get(id(b), []))
        common = set_a.intersection(set_b)
        if common:
            # single connector suffices
            for rid in common:
                bridge_alt[id(a)].append(rest_ids[rid])
                bridge_alt[id(b)].append(rest_ids[rid])
            continue
        # Else try chain up to N hops
        chain_ints = find_connector_chain(a, b, max_hops=3)
        if chain_ints:
            bridge_alt[id(a)].extend(chain_ints)
            bridge_alt[id(b)].extend(chain_ints)
    # Build shapes per seed and deduplicate
    all_shapes = []
    for seed in tqdm(main_intervals):
        # Merge proximity_alt with bridge_alt for this seed
        merged_alt = defaultdict(list)
        for item in proximity_alt.get(id(seed), []):
            merged_alt[id(seed)].append(item)
        for item in bridge_alt.get(id(seed), []):
            merged_alt[id(seed)].append(item)
        all_shapes.extend(shapes_for_interval(seed, proximity, merged_alt, snap_tol))
    shapes = _dedup_shapes(all_shapes, tol=snap_tol)
    return shapes

def _build_interval_graph(all_nodes: List[Interval], snap_tol: float) -> Tuple[Dict[int, List[int]], Dict[int, Interval]]:
    id_to_interval: Dict[int, Interval] = {id(iv): iv for iv in all_nodes}
    adj: Dict[int, List[int]] = defaultdict(list)
    nodes = list(id_to_interval.keys())
    for i in range(len(nodes)):
        idi = nodes[i]
        ai = id_to_interval[idi]
        for j in range(i + 1, len(nodes)):
            idj = nodes[j]
            aj = id_to_interval[idj]
            if compute_interval_distance(ai, aj, snap_tol) <= snap_tol:
                adj[idi].append(idj)
                adj[idj].append(idi)
    return adj, id_to_interval

dist_cache: Dict[Tuple[int, int], float] = {}

# Caches and helpers
def key(i1: Interval, i2: Interval) -> Tuple[int, int]:
    a, b = id(i1), id(i2)
    return (a, b) if a <= b else (b, a)


def is_adjacent(i1: Interval, i2: Interval, snap_tol) -> bool:
        k = key(i1, i2)
        if k in dist_cache:
            d = dist_cache[k]
        else:
            d = compute_interval_distance(i1, i2, snap_tol)
            dist_cache[k] = d
        return d <= snap_tol

def find_path_in_n_steps(
    interval_a: Interval,
    interval_b: Interval,
    all_intervals: List[Interval],
    snap_tol: float,
    n: int = 5,
) -> Optional[List[Interval]]:
    """
    Bounded BFS that computes adjacency on-the-fly: neighbors are intervals from
    all_intervals (plus the target) whose distance to the current interval is <= snap_tol.
    Uses a small cache to avoid repeated distance calculations.
    """
    print(len(dist_cache))
    if interval_a is interval_b:
        return [interval_a]

    from collections import deque

    # Candidate pool excludes the start; we add the target explicitly for adjacency checks
    candidates: List[Interval] = list(all_intervals)


    start_id = id(interval_a)
    target_id = id(interval_b)
    parent: Dict[int, Optional[int]] = {start_id: None}
    obj_by_id: Dict[int, Interval] = {start_id: interval_a, target_id: interval_b}

    q = deque([(interval_a, 0)])
    visited: Set[int] = {start_id}

    while q:
        cur_obj, depth = q.popleft()
        cur_id = id(cur_obj)

        # Early exit if directly adjacent to target
        if is_adjacent(cur_obj, interval_b, snap_tol):
            parent[target_id] = cur_id
            obj_by_id[target_id] = interval_b
            # reconstruct
            path_ids: List[int] = []
            x = target_id
            while x is not None:
                path_ids.append(x)
                x = parent.get(x)
            path_ids.reverse()
            return [obj_by_id[pid] for pid in path_ids]

        if depth >= n:
            continue
        # Explore neighbors among candidates
        for nb in candidates:
            nb_id = id(nb)
            if nb_id in visited:
                continue
            if not is_adjacent(cur_obj, nb, snap_tol):
                continue
            visited.add(nb_id)
            parent[nb_id] = cur_id
            obj_by_id[nb_id] = nb
            q.append((nb, depth + 1))
    return None


# def find_all_cycle_containing_interval(
#     seed: Interval,
#     intervals: List[Interval],
#     pair_paths: Dict[Tuple[int, int], List[Interval]],
#     max_len: int = 8,
# ) -> List[Schape]:
#     """
#     Treat pair_paths as a graph: for every path, connect consecutive intervals as undirected edges.
#     Enumerate simple cycles that include the seed interval and return them as closed shapes.
#     """
#     # Build node map
#     id_to_interval: Dict[int, Interval] = {id(iv): iv for iv in intervals}
#     id_to_interval[id(seed)] = seed

#     # Build adjacency from pair_paths
#     adj: Dict[int, List[int]] = defaultdict(list)
#     def add_edge(u: int, v: int):
#         if v not in adj[u]:
#             adj[u].append(v)
#         if u not in adj[v]:
#             adj[v].append(u)

#     for _, path in pair_paths.items():
#         if not path or len(path) < 2:
#             continue
#         # ensure all intervals are in the map
#         for iv in path:
#             id_to_interval[id(iv)] = iv
#         for i in range(len(path) - 1):
#             u = id(path[i])
#             v = id(path[i + 1])
#             add_edge(u, v)

#     seed_id = id(seed)
#     cycles: Set[Tuple[int, ...]] = set()
#     stack: List[int] = []

#     def dfs(curr: int, start: int, depth: int):
#         if depth > max_len:
#             return
#         stack.append(curr)
#         for nxt in adj.get(curr, []):
#             if nxt == start and depth >= 3:
#                 cyc = tuple(stack)
#                 # normalize cycle representation
#                 mins = min(cyc)
#                 idx = cyc.index(mins)
#                 rot1 = cyc[idx:] + cyc[:idx]
#                 rot2 = tuple(reversed(rot1))
#                 norm = tuple(rot1) if rot1 < rot2 else rot2
#                 cycles.add(norm)
#             elif nxt not in stack:
#                 dfs(nxt, start, depth + 1)
#         stack.pop()

#     dfs(seed_id, seed_id, 1)

#     results: List[Schape] = []
#     for cyc in cycles:
#         if seed_id not in cyc:
#             continue
#         intervals_in_cycle = [id_to_interval[iid] for iid in cyc if iid in id_to_interval]
#         if intervals_in_cycle:
#             results.append(
#                 Schape(kind="room", intervals=intervals_in_cycle, polygon=None, is_closed=True)
#             )
#     return results

def _id(interval):
    (x1, y1), (x2, y2) = interval.endpoints
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    if x1 == x2:
        n1 = min([(x1, y1), (x2, y2)], key=lambda x: x[1])
        n2 = max([(x1, y1), (x2, y2)], key=lambda x: x[1])
    else: 
        n1 = min([(x1, y1), (x2, y2)], key=lambda x: x[0])
        n2 = max([(x1, y1), (x2, y2)], key=lambda x: x[0])
    return f"{n1[0]}_{n1[1]}_{n2[0]}_{n2[1]}"


def find_all_cycle_containing_interval(
    seed: Interval,
    intervals: List[Interval],
    pair_paths: Dict[Tuple[int, int], List[Interval]],
    max_len: int = 8,
) -> List[Shape]:
    """
    Treat pair_paths as a graph: for every path, connect consecutive intervals as undirected edges.
    Enumerate simple cycles that include the seed interval and return them as closed shapes.
    """
    # Build node map and adjacency from provided pair_paths
    id_to_interval: Dict[int, Interval] = {_id(iv): iv for iv in intervals}
    adj: Dict[int, List[int]] = defaultdict(list)
    def add_edge(u: int, v: int):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    for _, path in pair_paths.items():
        if not path or len(path) < 2:
            continue
        for iv in path:
            id_to_interval[_id(iv)] = iv
        for i in range(len(path) - 1):
            u = _id(path[i])
            v = _id(path[i + 1])
            add_edge(u, v)

    seed_id = _id(seed)
    cycles: Set[Tuple[int, ...]] = set()
    stack: List[int] = []

    def dfs(curr: int, start: int, depth: int):
        if depth > max_len:
            return
        stack.append(curr)
        for nxt in adj.get(curr, []):
            if nxt == start and depth >= 3:
                cyc = tuple(stack)
                # normalize cycle representation
                # mins = min(cyc)
                # idx = cyc.index(mins)
                # rot1 = cyc[idx:] + cyc[:idx]
                # rot2 = tuple(reversed(rot1))
                # norm = tuple(rot1) if rot1 < rot2 else rot2
                print(cyc)
                cycles.add(cyc)
                break
            elif nxt not in stack:
                dfs(nxt, start, depth + 1)
        stack.pop()

    dfs(seed_id, seed_id, 1)

    results: List[Shape] = []
    for cyc in cycles:
        if seed_id not in cyc:
            continue
        intervals_in_cycle = [id_to_interval[iid] for iid in cyc if iid in id_to_interval]
        if intervals_in_cycle:
            corrected_intervals = connect_cycle(intervals_in_cycle, snap_tol  *3)
            if corrected_intervals is None:
                continue
            results.append(
                Shape(kind="room", intervals=corrected_intervals, polygon=None, is_closed=True)
            )
    selected_results = find_closest_cycles(seed, results)
    return selected_results

def _line_intersection(p1: Tuple[float, float], p2: Tuple[float, float], q1: Tuple[float, float], q2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    x1, y1 = p1; x2, y2 = p2; x3, y3 = q1; x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    return (float(px), float(py))


def _replace_endpoint(iv: Interval, endpoint_idx: int, new_pt: Tuple[float, float]) -> Interval:
    pts = [tuple(map(float, iv.endpoints[0])), tuple(map(float, iv.endpoints[1]))]
    pts[endpoint_idx] = (float(new_pt[0]), float(new_pt[1]))
    vec = np.array(pts[1]) - np.array(pts[0])
    angle = float(np.rad2deg(np.arctan2(vec[1], vec[0])))
    length = float(np.linalg.norm(vec))
    return Interval(
        endpoints=(pts[0], pts[1]),
        angle=angle,
        length=length,
        classification=iv.classification,
        thickness=iv.thickness,
        angle_label=iv.angle_label,
        segments=iv.segments or [iv],
    )


def connect_cycle(intervals: List[Interval], scale) -> List[Interval]:
    """
    Adjust a cycle so consecutive intervals meet cleanly:
    - Handle T-junction by trimming the interval being T-ed into at connection point.
    - If endpoints supposed to meet do not, set both to the intersection of their supporting lines.
    Returns corrected list preserving order.
    """
    if not intervals:
        return intervals

    corrected = [iv for iv in intervals]
    tol_dist = 2.0
    n = len(corrected)
    for i in range(n):
        a = corrected[i]
        b = corrected[(i + 1) % n]
        a0, a1 = tuple(a.endpoints[0]), tuple(a.endpoints[1])
        b0, b1 = tuple(b.endpoints[0]), tuple(b.endpoints[1])

        # Choose the endpoint of 'a' closest to segment 'b'
        proj_a0, t_a0, d_a0 = _project_point_to_segment(a0, b0, b1)
        proj_a1, t_a1, d_a1 = _project_point_to_segment(a1, b0, b1)
        if d_a0 <= d_a1:
            ea = 0; proj_a = proj_a0; ta = t_a0; da = d_a0
        else:
            ea = 1; proj_a = proj_a1; ta = t_a1; da = d_a1

        # Choose the endpoint of 'b' closest to segment 'a'
        proj_b0, t_b0, d_b0 = _project_point_to_segment(b0, a0, a1)
        proj_b1, t_b1, d_b1 = _project_point_to_segment(b1, a0, a1)
        if d_b0 <= d_b1:
            eb = 0; proj_b = proj_b0; tb = t_b0; db = d_b0
        else:
            eb = 1; proj_b = proj_b1; tb = t_b1; db = d_b1

        # T-junction handling: trim the interval being hit in the interior
        if da <= tol_dist and 1e-6 < ta < 1.0 - 1e-6:
            # a endpoint lands on interior of b -> trim b endpoint near proj_a to proj_a
            target_idx = 0 if np.linalg.norm(np.array(b0) - np.array(proj_a)) <= np.linalg.norm(np.array(b1) - np.array(proj_a)) else 1
            b = _replace_endpoint(b, target_idx, proj_a)
        if db <= tol_dist and 1e-6 < tb < 1.0 - 1e-6:
            target_idx = 0 if np.linalg.norm(np.array(a0) - np.array(proj_b)) <= np.linalg.norm(np.array(a1) - np.array(proj_b)) else 1
            a = _replace_endpoint(a, target_idx, proj_b)

        # Perfect the junction by intersecting supporting lines
        inter = _line_intersection(tuple(a.endpoints[0]), tuple(a.endpoints[1]), tuple(b.endpoints[0]), tuple(b.endpoints[1]))
        p11, p12, p21, p22, p = np.array(a.endpoints[0]), np.array(a.endpoints[1]), np.array(b.endpoints[0]), np.array(b.endpoints[1]), np.array(inter)
        n11, n12, n21, n22 = map(np.linalg.norm, [p11-p, p12-p, p21-p, p22-p])
        if min[n11, n12, n21, n22] > scale * 0.01:
            return None
        if inter is not None and np.isfinite(inter[0]) and np.isfinite(inter[1]):
            a = _replace_endpoint(a, ea, inter)
            b = _replace_endpoint(b, eb, inter)

        corrected[i] = a
        corrected[(i + 1) % n] = b

    return corrected


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _cycle_vertices(intervals: List[Interval]) -> List[Tuple[float, float]]:
    """Compute polygon vertices by intersecting consecutive interval lines."""
    n = len(intervals)
    verts: List[Tuple[float, float]] = []
    if n < 3:
        return verts
    for i in range(n):
        a = intervals[i]
        b = intervals[(i + 1) % n]
        inter = _line_intersection(tuple(a.endpoints[0]), tuple(a.endpoints[1]), tuple(b.endpoints[0]), tuple(b.endpoints[1]))
        if inter is None:
            # fallback to nearest endpoints
            cand = [a.endpoints[0], a.endpoints[1], b.endpoints[0], b.endpoints[1]]
            # pick the average of closest pair
            best = min(((p, q) for p in cand for q in cand), key=lambda pq: np.linalg.norm(np.array(pq[0]) - np.array(pq[1])))
            inter = tuple(((np.array(best[0]) + np.array(best[1])) / 2.0).tolist())
        verts.append((float(inter[0]), float(inter[1])))
    return verts


def _interval_direction(iv: Interval) -> np.ndarray:
    v = np.array(iv.endpoints[1]) - np.array(iv.endpoints[0])
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def find_closest_cycles(seed: Interval, cycles: List[Shape]) -> List[Shape]:
    """
    Choose the smallest-area cycles: one on the left and one on the right of the seed direction.
    If multiple, pick minimal absolute area. Seed mapping is tolerant to corrections.
    """
    if not cycles:
        return []
    # Determine seed direction possibly by matching similar interval inside cycles
    seed_dir = _interval_direction(seed)
    # Compute centroid of each cycle to decide side
    left_best = None
    right_best = None
    left_area = None
    right_area = None
    seed_origin = np.array(seed.endpoints[0])
    for shp in cycles:
        verts = shp.polygon if shp.polygon else _cycle_vertices(shp.intervals)
        if len(verts) < 3:
            continue
        area_signed = _polygon_area(verts)
        area = abs(area_signed)
        centroid = tuple(np.mean(np.array(verts), axis=0).tolist())
        vec_c = np.array(centroid) - seed_origin
        cross = seed_dir[0] * vec_c[1] - seed_dir[1] * vec_c[0]
        # categorize
        if cross >= 0:
            if left_best is None or area < left_area:
                left_best = shp; left_area = area
        else:
            if right_best is None or area < right_area:
                right_best = shp; right_area = area
        # store polygon back for drawing
        shp.polygon = verts
    out = []
    if left_best is not None:
        out.append(left_best)
    if right_best is not None:
        out.append(right_best)
    return out

def connect_main_lines_apporach_c(
    main_intervals: List[Interval],
    all_intervals: List[Interval],
    img_binary: Optional[np.ndarray],
    snap_tol: Optional[float] = None,
    n = 5,
) -> List[Shape]:
    # Configure tolerance
    if not main_intervals:
        return []
    if snap_tol is None:
        avg_len = float(np.mean([seg.length for seg in main_intervals])) if main_intervals else 10.0
        snap_tol = max(3.0, 0.02 * avg_len)
    
    # 1) Find shortest connector paths between pairs of mains (bounded by n)
    pair_paths: Dict[Tuple[int, int], List[Interval]] = {}
    for a, b in tqdm(product(main_intervals, main_intervals), total=len(main_intervals) ** 2):
        if a is b:
            continue
        path = find_path_in_n_steps(a, b, all_intervals, snap_tol=snap_tol, n=n)
        if path:
            pair_paths[(id(a), id(b))] = path

    # 2) For each main interval, find all cycles that include it using pair_paths graph
    all_shapes: List[Shape] = []
    for a in tqdm(main_intervals):
        cycles = find_all_cycle_containing_interval(a, intervals=(all_intervals or []) + main_intervals, pair_paths=pair_paths)
        all_shapes.extend(cycles)

    # 3) Add open shapes for mains not part of any returned cycle
    mains_in_cycles: Set[int] = set()
    for shp in all_shapes:
        for seg in shp.intervals:
            mains_in_cycles.add(id(seg))

    for a in main_intervals:
        if id(a) not in mains_in_cycles:
            all_shapes.append(Shape(kind="support", intervals=[a], polygon=None, is_closed=False))

    shapes = _dedup_shapes(all_shapes, tol=snap_tol)
    return shapes

    
    
    

def draw_shape(shapes: List[Shape], img_original: np.ndarray) -> np.ndarray:
    """
    Draw provided shapes onto the image. Closed shapes get distinct colors and polygon overlay;
    open shapes draw only their intervals.
    """
    if img_original is None or img_original.size == 0:
        return img_original
    if len(img_original.shape) == 2:
        canvas = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img_original.copy()

    rng = np.random.default_rng(12345)
    for shp in shapes:
        color = tuple(int(c) for c in rng.integers(50, 220, size=3).tolist())
        # Draw intervals
        for seg in shp.intervals:
            (x1, y1), (x2, y2) = seg.endpoints
            cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness=2, lineType=cv2.LINE_AA)
        # Draw polygon if closed
        if shp.is_closed and shp.polygon and len(shp.polygon) >= 3:
            pts = np.array([[int(round(x)), int(round(y))] for (x, y) in shp.polygon], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color=(color[0], color[1], color[2]))
            cv2.addWeighted(overlay, 0.1, canvas, 0.9, 0, canvas)
        # Draw assisting intervals in red
        for seg in shp.assisting_intervals:
            (x1, y1), (x2, y2) = seg.endpoints
            cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return canvas
