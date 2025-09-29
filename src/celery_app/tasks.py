import logging

from src.celery_app.app import celery_app
from src.core.pipeline import MainPipeline
from src.db.database import SessionLocal
from src.db import crud
from src.schemas.job import JobStatus


@celery_app.task
def digitalize_roof_plan_task(job_id: str, file_url: str):
    logging.info(f"Starting digitalization for job {job_id} with file {file_url}")
    db = SessionLocal()
    try:
        crud.update_job_status(db, job_id=job_id, status=JobStatus.IN_PROGRESS)

        result = MainPipeline().run(file_url=file_url, job_uuid=job_id)
        result_dict = result.dict()
        
        crud.update_job_status(db, job_id=job_id, status=JobStatus.SUCCESS, result=result_dict)
        logging.info(f"Successfully finished job {job_id}")
    except Exception as e:
        logging.error(f"Job {job_id} failed: {e}")
        crud.update_job_status(db, job_id=job_id, status=JobStatus.FAILURE, result={"error": str(e)})
    finally:
        db.close()