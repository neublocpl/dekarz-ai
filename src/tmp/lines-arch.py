# import fitz
# import cv2
# import numpy as np
# import math
#
# # ---------- Utility ----------
#
# def convert_pdf_to_image(page, zoom=2.0):
#     mat = fitz.Matrix(zoom, zoom)
#     pix = page.get_pixmap(matrix=mat, alpha=False)
#     arr = np.frombuffer(pix.samples, dtype=np.uint8)
#     if pix.n == 4:
#         img = arr.reshape(pix.height, pix.width, 4)
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
#     else:
#         img = arr.reshape(pix.height, pix.width, pix.n)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     return img.copy(), pix
#
# def scale_point(x, y, sx, sy, W, H):
#     xi = max(0, min(int(round(x * sx)), W - 1))
#     yi = max(0, min(int(round(y * sy)), H - 1))
#     return xi, yi
#
# def is_dashed(dashes):
#     if dashes is None:
#         return False
#     if isinstance(dashes, (list, tuple)):
#         return any(v > 0 for v in dashes)
#     return False
#
# def rgb_to_hsv255(color):
#     r, g, b = [int(c * 255) for c in color]
#     arr = np.uint8([[[r, g, b]]])
#     hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0][0]
#     return hsv  # (h,s,v)
#
# def classify_line(width, color, dashes,
#                   thin_threshold=0.4, thick_threshold=1.0):
#     if is_dashed(dashes):
#         return "dashed"
#
#     if color and isinstance(color, (tuple, list)) and len(color) >= 3:
#         hsv = rgb_to_hsv255(color)
#         h, s, v = hsv
#         if 20 <= h <= 40 and s > 100 and v > 150:
#             return "yellow"
#         if (h <= 10 or h >= 170) and s > 100:
#             return "red"
#
#     if width is None:
#         width = 1.0
#     try:
#         w = float(width)
#     except Exception:
#         w = 1.0
#     if w < thin_threshold:
#         return "thin"
#     elif w >= thick_threshold:
#         return "thick"
#
#     return "normal"
#
# def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=12, gap_len=6):
#     x1, y1 = pt1
#     x2, y2 = pt2
#     length = math.hypot(x2 - x1, y2 - y1)
#     if length <= 0:
#         return
#     dx = (x2 - x1) / length
#     dy = (y2 - y1) / length
#     seg = dash_len + gap_len
#     i = 0.0
#     while i < length:
#         start = i
#         end = min(i + dash_len, length)
#         xa = int(round(x1 + dx * start))
#         ya = int(round(y1 + dy * start))
#         xb = int(round(x1 + dx * end))
#         yb = int(round(y1 + dy * end))
#         cv2.line(img, (xa, ya), (xb, yb), color, thickness, lineType=cv2.LINE_AA)
#         i += seg
#
# def draw_line(img, p1, p2, line_type, thickness=1):
#     color_map = {
#         "thin": (0, 255, 0),
#         "thick": (0, 0, 255),
#         "dashed": (255, 0, 0),
#         "yellow": (0, 255, 255),
#         "red": (0, 0, 200),
#         "normal": (255, 0, 255),
#         "dashed_group": (0, 165, 255)
#     }
#     col = color_map.get(line_type, (255, 0, 255))
#     if line_type in ("dashed", "dashed_group"):
#         draw_dashed_line(img, p1, p2, col, thickness, dash_len=12, gap_len=6)
#     else:
#         cv2.line(img, p1, p2, col, thickness, lineType=cv2.LINE_AA)
#
# # ---------- Grupowanie i scalanie ----------
#
# def angle_of_line(p1, p2):
#     return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) % 180
#
# def line_params(p1, p2):
#     x1, y1 = p1
#     x2, y2 = p2
#     a = y1 - y2
#     b = x2 - x1
#     c = x1*y2 - x2*y1
#     norm = math.hypot(a, b)
#     if norm == 0:
#         return (0,0,0)
#     return a/norm, b/norm, c/norm
#
# def group_and_merge_dashed(lines, angle_tol=5, dist_tol=4, gap_tol=20):
#     """
#     Grupuje współliniowe segmenty w jedną linię przerywaną.
#     """
#     merged = []
#     used = set()
#
#     for i, li in enumerate(lines):
#         if i in used or li["type"] not in ("dashed", "thin", "normal"):
#             continue
#
#         group = [li]
#         ai = angle_of_line(li["p1"], li["p2"])
#         a1, b1, c1 = line_params(li["p1"], li["p2"])
#
#         for j, lj in enumerate(lines):
#             if j == i or j in used:
#                 continue
#             aj = angle_of_line(lj["p1"], lj["p2"])
#             if abs(ai - aj) > angle_tol:
#                 continue
#             a2, b2, c2 = line_params(lj["p1"], lj["p2"])
#             dist = abs(c1 - c2)
#             if dist > dist_tol:
#                 continue
#             # sprawdź odległość końców
#             d1 = min(math.hypot(li["p1"][0]-lj["p1"][0], li["p1"][1]-lj["p1"][1]),
#                      math.hypot(li["p2"][0]-lj["p2"][0], li["p2"][1]-lj["p2"][1]))
#             if d1 < gap_tol:
#                 group.append(lj)
#                 used.add(j)
#
#         if len(group) > 1:
#             # scalenie do najdalszych punktów
#             pts = [l["p1"] for l in group] + [l["p2"] for l in group]
#             xs, ys = zip(*pts)
#             pmin = (min(xs), min(ys))
#             pmax = (max(xs), max(ys))
#             merged.append({
#                 "p1": pmin, "p2": pmax,
#                 "width": 1.0,
#                 "color": None,
#                 "dashes": True,
#                 "type": "dashed_group",
#                 "thickness": max(l["thickness"] for l in group)
#             })
#             for g in group:
#                 used.add(lines.index(g))
#         else:
#             merged.append(li)
#             used.add(i)
#
#     # dodaj linie nieużyte
#     for i, l in enumerate(lines):
#         if i not in used:
#             merged.append(l)
#
#     return merged
#
# # ---------- Main analysis ----------
#
# def analyze_roof_plan(pdf_path, out_path="out.jpeg", zoom=2.0):
#     doc = fitz.open(pdf_path)
#     page = doc[0]
#
#     img, pix = convert_pdf_to_image(page, zoom)
#     H, W = img.shape[:2]
#     sx = pix.width / float(page.mediabox.width)
#     sy = pix.height / float(page.mediabox.height)
#
#     results = {"lines": [], "texts": []}
#
#     for drawing in page.get_drawings():
#         width = drawing.get("width", 1.0)
#         color = drawing.get("color", None)
#         dashes = drawing.get("dashes", None)
#
#         for item in drawing["items"]:
#             if item[0] != "l":
#                 continue
#             p1, p2 = item[1], item[2]
#             x1, y1 = scale_point(p1.x, p1.y, sx, sy, W, H)
#             x2, y2 = scale_point(p2.x, p2.y, sx, sy, W, H)
#
#             length_px = math.hypot(x2 - x1, y2 - y1)
#             if length_px < 6:
#                 continue
#
#             # filtracja krótkich grubych linii
#             if width is not None and width >= 1.0 and length_px < 15:
#                 continue
#
#             line_type = classify_line(width, color, dashes)
#             thickness_px = max(1, int(round((width or 1.0) * zoom)))
#
#             results["lines"].append({
#                 "p1": (x1, y1), "p2": (x2, y2),
#                 "width": width, "color": color,
#                 "dashes": dashes, "type": line_type,
#                 "thickness": thickness_px
#             })
#
#     # Grupowanie i scalanie przerywanych
#     results["lines"] = group_and_merge_segments(results["lines"])
#
#     # Rysowanie
#     for line in results["lines"]:
#         draw_line(img, line["p1"], line["p2"], line["type"], line["thickness"])
#
#     # Teksty
#     for block in page.get_text("dict")["blocks"]:
#         if "lines" not in block:
#             continue
#         for line in block["lines"]:
#             for span in line["spans"]:
#                 results["texts"].append(span["text"])
#
#     cv2.imwrite(out_path, img)
#     print(f"Zapisano wynik do: {out_path}")
#     return results
#
# def group_and_merge_segments(lines, angle_tol=5, dist_tol=4, gap_tol=20, rel_tol=0.2):
#     """
#     Grupuje współliniowe segmenty i rozstrzyga, czy są linią przerywaną, czy ciągłą.
#     Kryteria:
#       - dashed_group: segmenty i przerwy o podobnej długości (regularny wzór)
#       - continuous: segmenty różnej długości, bez powtarzalnych przerw
#     """
#     merged = []
#     used = set()
#
#     def angle_of_line(p1, p2):
#         return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) % 180
#
#     def line_params(p1, p2):
#         x1, y1 = p1
#         x2, y2 = p2
#         a = y1 - y2
#         b = x2 - x1
#         c = x1*y2 - x2*y1
#         norm = math.hypot(a, b)
#         if norm == 0:
#             return (0,0,0)
#         return a/norm, b/norm, c/norm
#
#     for i, li in enumerate(lines):
#         if i in used:
#             continue
#
#         group = [li]
#         ai = angle_of_line(li["p1"], li["p2"])
#         a1, b1, c1 = line_params(li["p1"], li["p2"])
#
#         for j, lj in enumerate(lines):
#             if j <= i or j in used:
#                 continue
#             aj = angle_of_line(lj["p1"], lj["p2"])
#             if abs(ai - aj) > angle_tol:
#                 continue
#             a2, b2, c2 = line_params(lj["p1"], lj["p2"])
#             dist = abs(c1 - c2)
#             if dist > dist_tol:
#                 continue
#             # sprawdzamy czy są blisko końcami
#             d1 = min(math.hypot(li["p1"][0]-lj["p1"][0], li["p1"][1]-lj["p1"][1]),
#                      math.hypot(li["p2"][0]-lj["p2"][0], li["p2"][1]-lj["p2"][1]))
#             if d1 < gap_tol:
#                 group.append(lj)
#                 used.add(j)
#
#         if len(group) > 1:
#             # Sprawdzamy czy to prawdziwa linia przerywana
#             seg_lengths = [math.hypot(l["p2"][0]-l["p1"][0], l["p2"][1]-l["p1"][1]) for l in group]
#             seg_lengths = [l for l in seg_lengths if l > 0]
#             if not seg_lengths:
#                 continue
#
#             mean_len = np.mean(seg_lengths)
#             rel_var = np.std(seg_lengths) / mean_len if mean_len > 0 else 0
#
#             # policz odległości między końcami
#             endpoints = []
#             for l in group:
#                 endpoints.extend([l["p1"], l["p2"]])
#             endpoints = sorted(endpoints, key=lambda p: (p[0], p[1]))
#             gaps = [math.hypot(endpoints[i+1][0]-endpoints[i][0], endpoints[i+1][1]-endpoints[i][1])
#                     for i in range(len(endpoints)-1)]
#             gaps = [g for g in gaps if g > 0]
#             mean_gap = np.mean(gaps) if gaps else 0
#             rel_gap_var = (np.std(gaps)/mean_gap) if mean_gap > 0 else 0
#
#             if rel_var < rel_tol and rel_gap_var < rel_tol and mean_gap > 2:
#                 # Prawdziwa linia przerywana
#                 pts = [l["p1"] for l in group] + [l["p2"] for l in group]
#                 xs, ys = zip(*pts)
#                 pmin = (min(xs), min(ys))
#                 pmax = (max(xs), max(ys))
#                 merged.append({
#                     "p1": pmin, "p2": pmax,
#                     "width": 1.0,
#                     "color": None,
#                     "dashes": True,
#                     "type": "dashed_group",
#                     "thickness": max(l["thickness"] for l in group)
#                 })
#             else:
# #                 # To linia ciągła, scalamy jako continuous
# #                 pts = [l["p1"] for l in group] + [l["p2"] for l in group]
# #                 xs, ys = zip(*pts)
# #                 pmin = (min(xs), min(ys))
# #                 pmax = (max(xs), max(ys))
# #                 merged.append({
# #                     "p1": pmin, "p2": pmax,
# #                     "width": li["width"],
# #                     "color": li["color"],
# #                     "dashes": None,
# #                     "type": "thin" if li["width"] < 1.0 else "thick",
# #                     "thickness": li["thickness"]
# #                 })
# #             for g in group:
# #                 used.add(lines.index(g))
# #         else:
# #             merged.append(li)
# #             used.add(i)
# #
# #     return merged
# #
# #
# # # ---------- Example run ----------
# #
# # # if __name__ == "__main__":
# # #     res = analyze_roof_plan("6.pdf", "out.jpeg", zoom=2.0)
# # #     print("Wykryto linii:", len(res["lines"]))
# # #     print("Przykładowe typy:", [l["type"] for l in res["lines"][:10]])
# # #     print("Przykładowe teksty:", res["texts"][:5])
#
#
# import fitz
# import cv2
# import numpy as np
# import math
#
# # ---------- Utilities ----------
#
# def convert_pdf_to_image(page, zoom=2.0):
#     mat = fitz.Matrix(zoom, zoom)
#     pix = page.get_pixmap(matrix=mat, alpha=False)
#     arr = np.frombuffer(pix.samples, dtype=np.uint8)
#     if pix.n == 4:
#         img = arr.reshape(pix.height, pix.width, 4)
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
#     else:
#         img = arr.reshape(pix.height, pix.width, pix.n)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     return img.copy(), pix
#
# def scale_point(x, y, sx, sy, W, H):
#     xi = max(0, min(int(round(x * sx)), W - 1))
#     yi = max(0, min(int(round(y * sy)), H - 1))
#     return xi, yi
#
# def is_dashed(dashes):
#     if dashes is None:
#         return False
#     if isinstance(dashes, (list, tuple)):
#         return any((v is not None and v > 0) for v in dashes)
#     return False
#
# def rgb_to_hsv255(color):
#     r, g, b = [int(max(0, min(1.0, c)) * 255) for c in color]
#     arr = np.uint8([[[r, g, b]]])
#     hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0][0]
#     return hsv  # (h,s,v)
#
# def classify_line(width, color, dashes,
#                   thin_threshold=0.4, thick_threshold=1.0):
#     # if explicit dashed pattern -> dashed
#     if is_dashed(dashes):
#         return "dashed"
#
#     # special colors (HSV)
#     if color and isinstance(color, (tuple, list)) and len(color) >= 3:
#         h, s, v = rgb_to_hsv255(color)
#         if 20 <= h <= 40 and s > 100 and v > 120:
#             return "yellow"
#         if (h <= 10 or h >= 170) and s > 100:
#             return "red"
#
#     # thickness-based
#     w = 1.0 if width is None else float(width)
#     if w < thin_threshold:
#         return "thin"
#     elif w >= thick_threshold:
#         return "thick"
#     else:
#         return "normal"
#
# def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=12, gap_len=6):
#     x1, y1 = pt1
#     x2, y2 = pt2
#     length = math.hypot(x2 - x1, y2 - y1)
#     if length <= 0:
#         return
#     dx = (x2 - x1) / length
#     dy = (y2 - y1) / length
#     seg = dash_len + gap_len
#     i = 0.0
#     while i < length:
#         start = i
#         end = min(i + dash_len, length)
#         xa = int(round(x1 + dx * start))
#         ya = int(round(y1 + dy * start))
#         xb = int(round(x1 + dx * end))
#         yb = int(round(y1 + dy * end))
#         cv2.line(img, (xa, ya), (xb, yb), color, thickness, lineType=cv2.LINE_AA)
#         i += seg
#
# def draw_line(img, p1, p2, line_type, thickness=1):
#     color_map = {
#         "thin": (0, 255, 0),
#         "thick": (0, 0, 255),
#         "dashed": (255, 0, 0),
#         "yellow": (0, 255, 255),
#         "red": (0, 0, 200),
#         "normal": (255, 0, 255),
#         "dashed_group": (0, 165, 255)
#     }
#     col = color_map.get(line_type, (255, 0, 255))
#     if line_type in ("dashed", "dashed_group"):
#         draw_dashed_line(img, p1, p2, col, thickness, dash_len=12, gap_len=6)
#     else:
#         cv2.line(img, p1, p2, col, thickness, lineType=cv2.LINE_AA)
#
# # ---------- Geometry helpers ----------
#
# def angle_of_line(p1, p2):
#     return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) % 180
#
# def unit_vector_and_ref(p1, p2):
#     dx = p2[0] - p1[0]
#     dy = p2[1] - p1[1]
#     L = math.hypot(dx, dy)
#     if L == 0:
#         return (1.0, 0.0), p1  # degenerate
#     ux, uy = dx / L, dy / L
#     return (ux, uy), p1
#
# def proj_t(point, pref, u):
#     # projection coordinate t of point relative to pref along unit vector u
#     return (point[0]-pref[0]) * u[0] + (point[1]-pref[1]) * u[1]
#
# # ---------- Group & merge logic (poprawiona) ----------
#
# def group_and_merge_segments(lines, angle_tol=5.0, dist_tol=4.0, gap_tol=20.0, rel_tol=0.25):
#     """
#     lines: lista dict z p1,p2,width,color,dashes,thickness,type
#     Zwraca listę linii po scaleniu grup współliniowych.
#     Decyzja: czy grupa to prawdziwa linia przerywana (regularne segmenty + równomierne przerwy)
#     czy raczej scalenie jako linia ciągła (pocięta na fragmenty).
#     Bezpieczne obsługiwanie width==None.
#     """
#     n = len(lines)
#     used = [False] * n
#     merged = []
#
#     for i in range(n):
#         if used[i]:
#             continue
#         li = lines[i]
#         # tylko rozważamy grupowanie dla linii, które mogą wchodzić w skład przerywanej
#         # (np. dashed/ thin / normal) - ale nie scalaj grubych krótkich elementów tutaj
#         if li.get("type") not in ("dashed", "thin", "normal"):
#             # zostaw takimi jakie są
#             merged.append(li)
#             used[i] = True
#             continue
#
#         # zaczynamy grupę od i
#         group_idx = [i]
#         ai = angle_of_line(li["p1"], li["p2"])
#         # parametry referencyjne dla prostej
#         (ux, uy), pref = unit_vector_and_ref(li["p1"], li["p2"])
#         # zbieramy kandydatów
#         for j in range(i+1, n):
#             if used[j]:
#                 continue
#             lj = lines[j]
#             aj = angle_of_line(lj["p1"], lj["p2"])
#             # kąt podobny?
#             if abs(ai - aj) > angle_tol:
#                 continue
#             # odległość prostych: oblicz odległość jednego punktu z lj do prostej li
#             # prosta li: p = pref + t*u ; distance = |( (p0 - pref) x u )|
#             p0 = lj["p1"]
#             cross = abs((p0[0]-pref[0]) * uy - (p0[1]-pref[1]) * ux)
#             if cross > dist_tol:
#                 continue
#             # dopuszczalny gap między końcami (luźne kryterium, dalsza selekcja later)
#             # przyjmujemy kandydatem
#             group_idx.append(j)
#
#         if len(group_idx) == 1:
#             merged.append(li)
#             used[i] = True
#             continue
#
#         # mamy grupę indeksów -> obliczamy statystyki długości segmentów i przerw
#         group_lines = [lines[k] for k in group_idx]
#
#         # oblicz proj t dla każdego segment i min/max t
#         seg_intervals = []
#         for l in group_lines:
#             t1 = proj_t(l["p1"], pref, (ux, uy))
#             t2 = proj_t(l["p2"], pref, (ux, uy))
#             seg_min = min(t1, t2)
#             seg_max = max(t1, t2)
#             seg_intervals.append((seg_min, seg_max, l))
#
#         # sortuj wg seg_min
#         seg_intervals.sort(key=lambda s: s[0])
#
#         # długości segmentów
#         seg_lengths = [s[1] - s[0] for s in seg_intervals if (s[1] - s[0]) > 0]
#         if not seg_lengths:
#             # nic sensownego
#             for k in group_idx:
#                 merged.append(lines[k])
#                 used[k] = True
#             continue
#
#         mean_seg = float(np.mean(seg_lengths))
#         std_seg = float(np.std(seg_lengths))
#         rel_var_seg = (std_seg / mean_seg) if mean_seg > 0 else 0.0
#
#         # przerwy: między kolejnymi segmentami (między max poprzedniego i min następnego)
#         gaps = []
#         for idx in range(len(seg_intervals)-1):
#             prev_max = seg_intervals[idx][1]
#             next_min = seg_intervals[idx+1][0]
#             gap = max(0.0, next_min - prev_max)
#             # ignoruj bardzo małe overlap jako zero
#             gaps.append(gap)
#         gaps_nonzero = [g for g in gaps if g > 0]
#         mean_gap = float(np.mean(gaps_nonzero)) if gaps_nonzero else 0.0
#         std_gap = float(np.std(gaps_nonzero)) if gaps_nonzero else 0.0
#         rel_var_gap = (std_gap / mean_gap) if mean_gap > 0 else 0.0
#
#         # Heurystyka: jeśli segmenty mają podobne długości (rel_var_seg < rel_tol)
#         # oraz przerwy są jednolite (rel_var_gap < rel_tol) i mean_gap jest sensowny (>2 px),
#         # to traktujemy jako prawdziwą przerywaną linię.
#         is_real_dashed = (rel_var_seg < rel_tol) and (rel_var_gap < rel_tol) and (mean_gap > 2.0)
#
#         # scalamy do jednego reprezentatywnego odcinka
#         overall_min = min(s[0] for s in seg_intervals)
#         overall_max = max(s[1] for s in seg_intervals)
#
#         # rekonstrukcja punktów końcowych w układzie XY (pref + u * t)
#         pmin = (pref[0] + ux * overall_min, pref[1] + uy * overall_min)
#         pmax = (pref[0] + ux * overall_max, pref[1] + uy * overall_max)
#
#         # ustal typ i szerokość
#         if is_real_dashed:
#             new_type = "dashed_group"
#             new_dashes = True
#         else:
#             # traktujemy jako ciągłą: wybierz typ w zależności od średniej szerokości
#             widths = [ (l.get("width") if l.get("width") is not None else 1.0) for l in group_lines ]
#             mean_w = float(np.mean([float(w) for w in widths]))
#             new_type = "thin" if mean_w < 1.0 else "thick"
#             new_dashes = None
#
#         new_thickness = int(max(1, round(max(l.get("thickness",1) for l in group_lines))))
#
#         merged.append({
#             "p1": (int(round(pmin[0])), int(round(pmin[1]))),
#             "p2": (int(round(pmax[0])), int(round(pmax[1]))),
#             "width": float(np.mean([ (l.get("width") if l.get("width") is not None else 1.0) for l in group_lines ])),
#             "color": None,
#             "dashes": new_dashes,
#             "type": new_type,
#             "thickness": new_thickness
#         })
#
#         # oznacz użyte indeksy
#         for k in group_idx:
#             used[k] = True
#
#     # dodaj pozostałe nieużyte (powinno ich nie być), ale dla pewności:
#     for i in range(n):
#         if not used[i]:
#             merged.append(lines[i])
#             used[i] = True
#
#     return merged
#
# # ---------- Main analysis (przykładowe użycie) ----------
#
# def analyze_roof_plan(pdf_path, out_path="out.jpeg", zoom=2.0):
#     doc = fitz.open(pdf_path)
#     page = doc[0]
#
#     img, pix = convert_pdf_to_image(page, zoom)
#     H, W = img.shape[:2]
#     sx = pix.width / float(page.mediabox.width)
#     sy = pix.height / float(page.mediabox.height)
#
#     results = {"lines": [], "texts": []}
#
#     for drawing in page.get_drawings():
#         width = drawing.get("width", 1.0)
#         color = drawing.get("color", None)
#         dashes = drawing.get("dashes", None)
#
#         for item in drawing["items"]:
#             if item[0] != "l":
#                 continue
#             p1, p2 = item[1], item[2]
#             x1, y1 = scale_point(p1.x, p1.y, sx, sy, W, H)
#             x2, y2 = scale_point(p2.x, p2.y, sx, sy, W, H)
#
#             length_px = math.hypot(x2 - x1, y2 - y1)
#             if length_px < 6:
#                 continue
#
#             # filtracja krótkich grubych linii (usuwamy fragmenty liter itp.)
#             if width is not None and float(width) >= 1.0 and length_px < 15:
#                 continue
#
#             line_type = classify_line(width, color, dashes)
#             thickness_px = max(1, int(round((width or 1.0) * zoom)))
#
#             results["lines"].append({
#                 "p1": (x1, y1), "p2": (x2, y2),
#                 "width": width, "color": color,
#                 "dashes": dashes, "type": line_type,
#                 "thickness": thickness_px
#             })
#
#     # Grupowanie i scalanie (poprawiona funkcja)
#     results["lines"] = group_and_merge_segments(results["lines"],
#                                                angle_tol=5.0,
#                                                dist_tol=4.0,
#                                                gap_tol=20.0,
#                                                rel_tol=0.25)
#
#     # Rysowanie
#     for line in results["lines"]:
#         draw_line(img, line["p1"], line["p2"], line["type"], line["thickness"])
#
#     # Teksty (wektorowe)
#     for block in page.get_text("dict")["blocks"]:
#         if "lines" not in block:
#             continue
#         for line in block["lines"]:
#             for span in line["spans"]:
#                 results["texts"].append(span.get("text", ""))
#
#     cv2.imwrite(out_path, img)
#     print(f"Zapisano wynik do: {out_path}")
#     return results
#
# # # ---------- Example run ----------
# # if __name__ == "__main__":
# #     res = analyze_roof_plan("6.pdf", "out.jpeg", zoom=2.0)
# #     print("Wykryto linii:", len(res["lines"]))
# #     print("Przykładowe typy:", [l["type"] for l in res["lines"][:20]])
# #     print("Wykryto tekstów:", len(res["texts"]))


# import fitz
# import cv2
# import numpy as np
# import math
#
# # ---------- Konfiguracja ----------
# THIN_THRESHOLD = 0.3
# THICK_THRESHOLD = 0.8
# MIN_LINE_LEN = 6
# MIN_THICK_LEN = 20  # minimalna długość dla grubych
# MERGE_ENABLED_TYPES = ("normal", "dashed")  # tylko te scalane
#
# COLOR_MAP = {
#     "thin": (0, 255, 0),
#     "thick": (0, 0, 255),
#     "normal": (255, 0, 255),
#     "yellow": (0, 255, 255),
#     "red": (0, 0, 200),
#     "dashed": (255, 0, 0),
#     "dashed_group": (0, 165, 255),
# }
#
# # ---------- Utility ----------
# def convert_pdf_to_image(page, zoom=2.0):
#     pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
#     arr = np.frombuffer(pix.samples, dtype=np.uint8)
#     if pix.n == 4:
#         img = arr.reshape(pix.height, pix.width, 4)
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
#     else:
#         img = arr.reshape(pix.height, pix.width, pix.n)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     return img.copy(), pix
#
# def scale_point(x, y, sx, sy, W, H):
#     xi = max(0, min(int(round(x * sx)), W - 1))
#     yi = max(0, min(int(round(y * sy)), H - 1))
#     return xi, yi
#
# def rgb_to_hsv255(color):
#     r, g, b = [int(max(0, min(1.0, c)) * 255) for c in color]
#     arr = np.uint8([[[r, g, b]]])
#     hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0][0]
#     return hsv  # (h,s,v)
#
# # ---------- Klasyfikacja ----------
# def classify_line_type(width, color, dashes):
#     w = max(0.01, float(width) if width is not None else 1.0)
#
#     # kolor żółty/czerwony
#     if color and isinstance(color, (tuple, list)) and len(color) >= 3:
#         h, s, v = rgb_to_hsv255(color)
#         if 20 <= h <= 40 and s > 100 and v > 120:
#             return "yellow"
#         if (h <= 10 or h >= 170) and s > 100:
#             return "red"
#
#     # dash — bezpieczna obsługa
#     if dashes:
#         try:
#             numeric_dashes = [float(val) for val in dashes if isinstance(val, (int, float))]
#             if any(val > 0 for val in numeric_dashes):
#                 return "dashed"
#         except Exception:
#             pass  # jeśli dashes jest dziwne (np. string) → ignorujemy
#
#     # grubość
#     if w < THIN_THRESHOLD:
#         return "thin"
#     elif w >= THICK_THRESHOLD:
#         return "thick"
#     else:
#         return "normal"
#
# # ---------- Ekstrakcja ----------
# def extract_lines_from_pdf(page, sx, sy, W, H, zoom=2.0):
#     lines = []
#     for drawing in page.get_drawings():
#         print(drawing)
#         width = drawing.get("width", 1.0)
#         color = drawing.get("color", None)
#         dashes = drawing.get("dashes", None)
#         for item in drawing["items"]:
#             if item[0] != "l":
#                 continue
#
#             print(len(item))
#             p1, p2 = item[1], item[2]
#             x1, y1 = scale_point(p1.x, p1.y, sx, sy, W, H)
#             x2, y2 = scale_point(p2.x, p2.y, sx, sy, W, H)
#             length_px = math.hypot(x2 - x1, y2 - y1)
#             lines.append({
#                 "p1": (x1, y1),
#                 "p2": (x2, y2),
#                 "width": width,
#                 "color": color,
#                 "dashes": dashes,
#                 "length": length_px,
#                 "thickness": max(1, int(round((width or 1.0) * zoom)))
#             })
#     return lines
#
# # ---------- Filtrowanie ----------
# # def filter_lines(lines):
# #     filtered = []
# #     for l in lines:
# #         if l["length"] < MIN_LINE_LEN:
# #             continue
# #         if l["width"] and l["width"] >= THICK_THRESHOLD and l["length"] < MIN_THICK_LEN:
# #             continue  # odrzuć krótkie grube
# #         filtered.append(l)
# #     return filtered
#
# # ---------- Grupowanie ----------
# def merge_collinear_segments(lines):
#     # Na razie nie scalaj "thin"
#     # TODO: tutaj można dodać prostsze scalanie, bez przedłużania
#     return lines
#
# # ---------- Render ----------
# def render_lines(img, lines):
#     for l in lines:
#         line_type = l["type"]
#         col = COLOR_MAP.get(line_type, (255, 0, 255))
#         cv2.line(img, l["p1"], l["p2"], col, l["thickness"], lineType=cv2.LINE_AA)
#     return img
#
# # ---------- Main ----------
# def analyze_roof_plan(img, pix, page, out_path="out.jpeg", zoom=2.0):
#     H, W = img.shape[:2]
#     sx = pix.width / float(page.mediabox.width)
#     sy = pix.height / float(page.mediabox.height)
#
#     # ekstrakcja
#     lines = extract_lines_from_pdf(page, sx, sy, W, H, zoom)
#     # filtrowanie
#     lines = filter_lines(lines)
#     # klasyfikacja
#     for l in lines:
#         l["type"] = classify_line_type(l["width"], l["color"], l["dashes"])
#     # scalanie tylko dla normalnych
#     lines = merge_normal_segments(lines)
#     # render
#     img = render_lines(img, lines)
#     cv2.imwrite(out_path, img)
#
#     return {"lines": lines}
#
# def filter_lines(lines):
#     filtered = []
#     for l in lines:
#         length = l["length"]
#         width = l["width"] if l["width"] is not None else 1.0
#         t = classify_line_type(width, l["color"], l["dashes"])
#
#         if length < MIN_LINE_LEN:
#             continue
#
#         # dodatkowa filtracja grubych / czerwonych
#         if t in ("thick", "red"):
#             if length < MIN_THICK_LEN:
#                 continue
#             # odrzuć jeśli długość nieproporcjonalna do grubości
#             if length / max(width, 0.1) < 5:
#                 continue
#
#         filtered.append(l)
#     return filtered
#
# def merge_normal_segments(lines, angle_tol=5, dist_tol=3, gap_tol=20, rel_tol=0.25):
#     """
#     Łączy współliniowe segmenty typu 'normal' i rozstrzyga czy to 'dashed_group'
#     czy pojedyncza linia ciągła.
#     """
#     merged = []
#     used = [False] * len(lines)
#
#     def angle(p1, p2):
#         return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) % 180
#
#     for i, li in enumerate(lines):
#         if used[i] or li["type"] != "normal":
#             continue
#
#         group = [i]
#         ai = angle(li["p1"], li["p2"])
#
#         for j, lj in enumerate(lines):
#             if j <= i or used[j] or lj["type"] != "normal":
#                 continue
#             aj = angle(lj["p1"], lj["p2"])
#             if abs(ai - aj) > angle_tol:
#                 continue
#             # odległość między końcami
#             d1 = min(math.hypot(li["p1"][0]-lj["p1"][0], li["p1"][1]-lj["p1"][1]),
#                      math.hypot(li["p2"][0]-lj["p2"][0], li["p2"][1]-lj["p2"][1]))
#             if d1 < gap_tol:
#                 group.append(j)
#
#         if len(group) == 1:
#             merged.append(li)
#             used[i] = True
#             continue
#
#         # policz długości segmentów i przerwy
#         seg_lengths = [lines[k]["length"] for k in group]
#         mean_len = np.mean(seg_lengths)
#         rel_var_len = np.std(seg_lengths) / mean_len if mean_len > 0 else 0
#
#         # policz przerwy między końcami
#         endpoints = []
#         for k in group:
#             endpoints.extend([lines[k]["p1"], lines[k]["p2"]])
#         endpoints = sorted(endpoints, key=lambda p: (p[0], p[1]))
#         gaps = [math.hypot(endpoints[idx+1][0]-endpoints[idx][0],
#                            endpoints[idx+1][1]-endpoints[idx][1])
#                 for idx in range(len(endpoints)-1)]
#         gaps_nonzero = [g for g in gaps if g > 0]
#         mean_gap = np.mean(gaps_nonzero) if gaps_nonzero else 0
#         rel_var_gap = (np.std(gaps_nonzero)/mean_gap) if mean_gap > 0 else 0
#
#         # scalanie
#         if rel_var_len < rel_tol and rel_var_gap < rel_tol and mean_gap > 2:
#             line_type = "dashed_group"
#         else:
#             line_type = "normal"
#
#         xs = [pt[0] for k in group for pt in (lines[k]["p1"], lines[k]["p2"])]
#         ys = [pt[1] for k in group for pt in (lines[k]["p1"], lines[k]["p2"])]
#         merged.append({
#             "p1": (min(xs), min(ys)),
#             "p2": (max(xs), max(ys)),
#             "width": np.mean([lines[k]["width"] for k in group if lines[k]["width"]]),
#             "color": None,
#             "dashes": None,
#             "type": line_type,
#             "thickness": max(l["thickness"] for k,l in enumerate(lines) if k in group)
#         })
#
#         for k in group:
#             used[k] = True
#
#     # dodaj resztę
#     for i, li in enumerate(lines):
#         if not used[i]:
#             merged.append(li)
#
#     return merged
#
#
# # ---------- Example run ----------
# # if __name__ == "__main__":
# #     res = analyze_roof_plan("6.pdf", "out.jpeg", zoom=2.0)
# #     print("Wykryto linii:", len(res["lines"]))
# #     print("Przykładowe typy:", [l["type"] for l in res["lines"][:20]])
#
# import cv2
# import numpy as np
#
# def remove_thin(image, lines, inpaint_radius=2):
#     """
#     Usuwa linie typu 'thin' z obrazu za pomocą inpaintingu.
#     image: np.ndarray (oryginalny obraz z liniami narysowanymi)
#     lines: lista wykrytych linii (dict z analyze_roof_plan)
#     """
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#
#     for l in lines:
#         if l["type"] == "thin":
#             cv2.line(mask, l["p1"], l["p2"], 255, max(1, 2 * l["thickness"]))
#
#     # Inpainting - wypełnia maskowane fragmenty otoczeniem
#     cleared = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
#     return cleared

import fitz
import cv2
import numpy as np
import math

# ---------- Libraries to Install ----------
# pip install PyMuPDF opencv-python numpy

# ---------- Configuration ----------
# Dynamic thresholding parameters
BLOCK_SIZE = 11  # Odd number, neighborhood size
C_CONSTANT = 2  # Constant subtracted from the mean

# Hough Line Transform parameters
HOUGH_RHO = 1  # Distance resolution of the accumulator in pixels
HOUGH_THETA = np.pi / 180  # Angle resolution of the accumulator in radians
HOUGH_THRESHOLD = 50  # Minimum number of intersections to detect a line
HOUGH_MIN_LINE_LENGTH = 30  # Minimum line length
HOUGH_MAX_LINE_GAP = 10  # Maximum gap between line segments to be considered a single line

# Line classification and filtering
MIN_LINE_LEN_PX = 6
THIN_THICKNESS_THRESHOLD = 2  # Thickness in pixels
THICK_THICKNESS_THRESHOLD = 5  # Thickness in pixels
MIN_THICK_LEN_PX = 20

COLOR_MAP = {
    "thin": (0, 255, 0),
    "thick": (0, 0, 255),
    "normal": (255, 0, 255),
    "yellow": (0, 255, 255),  # Note: color detection is complex from images
    "red": (0, 0, 200),
    "dashed": (255, 0, 0),
    "dashed_group": (0, 165, 255),
}


# ---------- Utility ----------
def convert_to_grayscale_image(file_path):
    """
    Reads an image file (PDF, JPG, PNG) and converts it to a grayscale numpy array.
    Handles PDFs by rendering the first page.
    """
    try:
        # Check if the file is a PDF
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = img.reshape(pix.height, pix.width)  # Grayscale already
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            return gray, img

        # Assume it's a standard image file
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"File not found or could not be opened: {file_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray, img
    except Exception as e:
        print(f"Error converting file to image: {e}")
        return None, None


def classify_line_type(thickness):
    """
    Classifies a line based on its detected pixel thickness.
    """
    if thickness <= THIN_THICKNESS_THRESHOLD:
        return "thin"
    elif thickness >= THICK_THICKNESS_THRESHOLD:
        return "thick"
    else:
        return "normal"


# ---------- Main Functions ----------
def get_adaptive_threshold(gray_img):
    """
    Applies adaptive thresholding to an image.
    This dynamically calculates the threshold for small regions.
    """
    return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, BLOCK_SIZE, C_CONSTANT)


def extract_lines_from_image(thresh_img):
    """
    Detects lines in a binary image using the Probabilistic Hough Line Transform.
    Returns a list of dictionaries with line properties.
    """
    lines_p = cv2.HoughLinesP(thresh_img, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                              minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)

    if lines_p is None:
        return []

    lines = []
    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        length_px = math.hypot(x2 - x1, y2 - y1)

        # Calculate approximate thickness (can be a more complex function)
        # For simplicity, we'll assume a fixed thickness based on the transform,
        # or you can estimate based on local pixel analysis.
        # A more advanced approach would measure the width of the line in the original image.
        thickness = 1

        lines.append({
            "p1": (x1, y1),
            "p2": (x2, y2),
            "length": length_px,
            "thickness": thickness
        })
    return lines


def filter_and_classify_lines(lines):
    """
    Filters and classifies detected lines.
    """
    filtered = []
    for l in lines:
        if l["length"] < MIN_LINE_LEN_PX:
            continue

        l["type"] = classify_line_type(l["thickness"])

        if l["type"] == "thick" and l["length"] < MIN_THICK_LEN_PX:
            continue

        filtered.append(l)
    return filtered


def render_lines(img, lines):
    """
    Renders the detected lines on a copy of the original image.
    """
    display_img = img.copy()
    for l in lines:
        line_type = l["type"]
        color = COLOR_MAP.get(line_type, (255, 0, 255))
        # Draw the line with detected properties
        cv2.line(display_img, l["p1"], l["p2"], color, l["thickness"], lineType=cv2.LINE_AA)
    return display_img


def remove_lines_with_inpainting(image, lines, line_type_to_remove="thin", inpaint_radius=3):
    """
    Removes lines of a specific type from an image using inpainting.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for l in lines:
        if l["type"] == line_type_to_remove:
            cv2.line(mask, l["p1"], l["p2"], 255, max(1, l["thickness"] + inpaint_radius))

    cleared_image = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleared_image


# ---------- Main Execution Flow ----------
def analyze_image_file(file_path, out_path_lines="out_lines.jpeg", out_path_removed="out_no_thin.jpeg"):
    """
    Main function to analyze any graphical file.
    """
    gray_img, original_img = convert_to_grayscale_image(file_path)
    if gray_img is None:
        return {"error": "Failed to load image."}

    # Step 1: Pre-processing with dynamic thresholding
    thresh_img = get_adaptive_threshold(gray_img)

    # Step 2: Line detection
    detected_lines = extract_lines_from_image(thresh_img)

    # Step 3: Classification and filtering
    classified_lines = filter_and_classify_lines(detected_lines)

    # Step 4: Render lines for visualization
    output_img_lines = render_lines(original_img, classified_lines)
    cv2.imwrite(out_path_lines, output_img_lines)
    print(f"Detected lines rendered and saved to {out_path_lines}")

    # Step 5: Remove thin lines and save a new image
    img_without_thin_lines = remove_lines_with_inpainting(original_img, classified_lines, "thin")
    cv2.imwrite(out_path_removed, img_without_thin_lines)
    print(f"Image with thin lines removed saved to {out_path_removed}")

    return {"lines": classified_lines}


# Example usage
if __name__ == "__main__":
    # file_to_process = "your_image_file.jpg"  # Or "your_pdf_file.pdf"
    file_to_process = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow/6.pdf"
    results = analyze_image_file(file_to_process)
    if "error" not in results:
        print(f"Detected {len(results['lines'])} lines.")

# if __name__ == "__main__":
#
#     # out = analyze_roof_plan("/Users/grzegorzstermedia/Downloads/Rzuty_dachow/6.pdf", "out.jpeg")
#     pdf_path = "/Users/grzegorzstermedia/Downloads/Rzuty_dachow/6.pdf"
#
#     _doc = fitz.open(pdf_path)
#     _page = _doc[0]
#     _img, _pix = convert_pdf_to_image(_page, zoom=2.0)
#
#     res = analyze_roof_plan(_img, _pix, _page, "out.jpeg", zoom=2.0)
#     print("Wykryto linii:", len(res["lines"]))
#     print("Przykładowe typy:", [l["type"] for l in res["lines"][:10]])
#
#     # _doc = fitz.open(pdf_path)
#     # _page = _doc[0]
#     clear_img = remove_thin(_img, res["lines"], 1)
#     clear_img[clear_img > 200] = 255
#     cv2.imwrite("out_no_thin.jpeg", clear_img)
#     print(_pix)
#     res = analyze_roof_plan(clear_img, _pix, _page, "out2.jpeg", zoom=2.0)
#     print("Obraz bez cienkich linii zapisany do out_no_thin.jpeg")
#     # print("Przykładowe teksty:", res["texts"][:5])
