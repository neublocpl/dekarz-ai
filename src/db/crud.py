import uuid
from sqlalchemy.orm import Session

from src.db import models
from src.schemas import job as job_schema


def create_job(db: Session) -> models.Job:
    db_job = models.Job()
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def get_job(db: Session, job_id: uuid.UUID) -> models.Job | None:
    return db.query(models.Job).filter(models.Job.id == job_id).first()


def update_job_status(db: Session, job_id: uuid.UUID, status: job_schema.JobStatus, result: dict = None):
    db_job = get_job(db, job_id)
    if db_job:
        db_job.status = status
        if result:
            db_job.result = result
        db.commit()
        db.refresh(db_job)
    return db_job