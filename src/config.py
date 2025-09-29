import os
from pathlib import Path

APP_VERSION = "0.1.0"

# --- Postgres ---
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"postgres:5432/{POSTGRES_DB}"
)

# --- Celery ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")

# --- File Storage ---
DATA_DIR = Path("/app/data")

INPUT_FILES_DIR = DATA_DIR / "input"
TEST_FILES_DIR = DATA_DIR / "test"
DEBUG_FILES_DIR = DATA_DIR / "debug"
MODEL_FILES_DIR = DATA_DIR / "models"

STATIC_FILES_DIR = {
    "input": INPUT_FILES_DIR,
    "test": TEST_FILES_DIR,
    "debug": DEBUG_FILES_DIR,
}


# --- AI Model ---

MODELS_DIR = Path("/app/models")
UNNECESESARY_ELEMENTS_MODEL_PATH = MODELS_DIR / "task_01_rcnn.pth"
