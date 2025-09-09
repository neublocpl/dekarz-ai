from fastapi import FastAPI

from src.routers import health, transform


app = FastAPI(
    title="Dekarz AI",
    description="API for Dekarz AI application",
    version="0.1.0",
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(transform.router, prefix="/transform", tags=["Transform"])


@app.get("/")
async def root():
    return {"message": "Welcome to the Dekarz AI API"}