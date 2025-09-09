from fastapi import APIRouter, status


router = APIRouter()


@router.post("", summary="Digitalize roof plan (create 3D model data)")
async def digitalize_roof_plan():
    return {"status": "ok"}


@router.post("/test_all", summary="Process all test cases")
async def test_all():
    return {"status": "ok"}