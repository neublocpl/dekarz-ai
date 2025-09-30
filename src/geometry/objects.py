import math
from typing import List, Optional, Tuple
from enum import Enum
import cv2
import numpy as np
from pydantic import BaseModel, Field

class TextType(str, Enum):
    """Enumeration for the type of recognized text."""
    NUMERICAL = "numerical"
    ANGLE = "angle"
    PERCENTAGE = "percentage"
    NATURAL_TEXT = "natural_text"

class TextStructure(str, Enum):
    """Enumeration for the structural layout of the text."""
    SINGLE_LINE = "single_line"
    MULTI_LINE = "multi_line"

class TextData(BaseModel):
    """
    Represents a piece of recognized text and its properties.
    """
    bounding_box: List[Tuple[float, float]] = Field(
        ...,
        description="A list of (x, y) coordinates for the text's bounding box polygon."
    )
    text: str = Field(..., description="The recognized text string.")
    orientation: float = Field(..., description="The orientation angle of the text in degrees.")
    text_type: TextType = Field(..., description="The classified type of the text content.")
    structure: TextStructure = Field(..., description="The structural layout of the text.")

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates the geometric center of the bounding box."""
        if not self.bounding_box:
            return (0.0, 0.0)
        sum_x = sum(p[0] for p in self.bounding_box[2:])
        sum_y = sum(p[1] for p in self.bounding_box[2:])
        num_points = len(self.bounding_box[2:])
        return (sum_x / num_points, sum_y / num_points)


class Interval(BaseModel):
    """
    Represents a detected line segment and its properties.

    The obligatory fields (endpoints, angle, length) are populated during
    detection. The optional fields (classification, thickness) are populated
    during a separate classification step.
    """

    # --- Obligatory fields (from detection) ---
    endpoints: Tuple[
        Tuple[int | float, int | float], Tuple[int | float, int | float]
    ] = Field(..., description="The (x1, y1) and (x2, y2) coordinates of the line.")
    angle: float = Field(..., description="The angle of the line in degrees.")
    length: float = Field(..., description="The length of the line in pixels.")

    # --- Optional fields (from classification) ---
    classification: Optional[str] = Field(
        None, description="Category of the line (e.g., 'thin', 'thick', 'yellow')."
    )
    thickness: Optional[float] = Field(
        None, description="The calculated thickness of the line in pixels."
    )
    angle_label: Optional[int] = Field(None)

    # --- Optional fields (for post-processing/merging) ---
    segments: Optional[List["Interval"]] = Field(
        default=None,
        description="Intervals of original segments merged into this interval.",
    )
    assigned_texts: Optional[List[TextData]] = Field(
        default_factory=list,
        description="A list of text items assigned to this interval."
    )

Interval.model_rebuild()


