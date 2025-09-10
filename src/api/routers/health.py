from fastapi import APIRouter, status


router = APIRouter()


@router.get("", response_model=dict, status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}