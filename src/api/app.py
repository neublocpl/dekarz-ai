import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src import config
from src.api.routers import health, transform

logging.basicConfig(format="[%(asctime)s] - %(name)s - %(message)s", level=logging.INFO)


for directory in config.STATIC_FILES_DIR.values():
    directory.mkdir(exist_ok=True)

app = FastAPI(
    title="Dekarz AI",
    description="API for Dekarz AI application",
    version=config.APP_VERSION,
)

app.mount("/static", StaticFiles(directory=config.DATA_DIR), name="static_files")

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(transform.router, prefix="/transform", tags=["Transform"])


@app.get("/")
async def root():
    return {"message": "Welcome to the Dekarz AI API"}