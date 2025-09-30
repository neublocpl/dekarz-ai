from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import math
from src.geometry.objects import Interval, TextData
import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_text(image, texts):
    output_img = image.copy()
    for text_data in texts:
        box = text_data.bounding_box
        text = text_data.text
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(output_img, [pts], True, (0, 255, 0), 2)
        cv2.putText(output_img, text, tuple(map(int, box[0])), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)

        center_point = tuple(map(int, text_data.center))
        cv2.circle(output_img, center_point, 5, (255, 0, 0), -1) 

    plt.figure(figsize=(30, 30))
    plt.imshow(output_img)
    plt.show()


def _distance_point_to_segment(
    point: Tuple[float, float],
    segment_p1: Tuple[float, float],
    segment_p2: Tuple[float, float],
) -> float:
    """Calculates the shortest distance from a point to a line segment."""
    px, py = point
    x1, y1 = segment_p1
    x2, y2 = segment_p2

    dx, dy = x2 - x1, y2 - y1
    
    # If the segment is just a point
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    # Calculate the projection of the point onto the line
    t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)

    # Clamp t to the [0, 1] range to stay on the segment
    t = max(0, min(1, t))

    # Find the coordinates of the closest point on the segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Return the distance to that point
    return math.hypot(px - closest_x, py - closest_y)


def assign_text_to_intervals(
    lines: List[Interval], texts: List[TextData]
) -> List[Interval]:
    """
    Assigns each text item to the geometrically closest line interval.

    Args:
        lines: A list of Interval objects.
        texts: A list of TextData objects to be assigned.

    Returns:
        The list of Interval objects with the `assigned_texts` field populated.
    """
    if not lines or not texts:
        return lines

    # Ensure the assigned_texts list is empty before starting
    for line in lines:
        line.assigned_texts = []

    # For each text, find the closest line and assign it
    for text in texts:
        text_center = text.center
        
        min_distance = float('inf')
        closest_line = None

        for line in lines:
            distance = _distance_point_to_segment(
                text_center, line.endpoints[0], line.endpoints[1]
            )
            if distance < min_distance:
                min_distance = distance
                closest_line = line
        
        if closest_line:
            closest_line.assigned_texts.append(text)
            
    return lines