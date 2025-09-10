import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import HttpUrl

from src.db import crud
from src.db.database import get_db
from src.celery_app.tasks import digitalize_roof_plan_task
from src.schemas import job as job_schema

router = APIRouter()


@router.post("", summary="Digitalize roof plan (create 3D model data)")
async def digitalize_roof_plan(file_url: HttpUrl, db: Session = Depends(get_db)):
    job = crud.create_job(db)
    if not job:
        raise HTTPException(status_code=500, detail="Could not create job.")

    digitalize_roof_plan_task.delay(job_id=str(job.id), file_url=str(file_url))

    return {"job_id": job.id, "status": job.status}


@router.get("/{job_id}", response_model=job_schema.Job, summary="Get job status and result")
async def get_job_status(job_id: uuid.UUID, db: Session = Depends(get_db)):
    db_job = crud.get_job(db, job_id=job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return db_job


# @router.post("/test_all", summary="Process all test cases")
# async def test_all():
#     return {"status": "ok"}