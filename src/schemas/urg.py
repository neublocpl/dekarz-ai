from pydantic import BaseModel, Field
from typing import Optional


class URG(BaseModel):
    points: Optional[list["Point"]] = Field(default_factory=list)
    edges: Optional[list["Edge"]] = Field(default_factory=list)
    polygons: Optional[list["Polygon"]] = Field(default_factory=list)


class Point(BaseModel):
    id: int = Field(..., description="Unique identifier for the point")
    image_x: int = Field(..., description="X coordinate in the image")
    image_y: int = Field(..., description="Y coordinate in the image")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")


class Edge(BaseModel):
    id: int = Field(..., description="Unique identifier for the edge")
    start_id: int = Field(..., description="ID of the start point")
    end_id: int = Field(..., description="ID of the end point")


class Polygon(BaseModel):
    id: int = Field(..., description="Unique identifier for the polygon")
    edges: list[int] = Field(..., description="List of edge IDs that form the polygon")