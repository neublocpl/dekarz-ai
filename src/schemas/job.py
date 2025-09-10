import uuid
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class JobBase(BaseModel):
    status: JobStatus = Field(default=JobStatus.PENDING)
    result: dict | None = None


class JobCreate(JobBase):
    pass


class Job(JobBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        orm_mode = True