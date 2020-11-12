from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Body, File, UploadFile

from app.config import IMAGE_DIRECTORY
from app.core.db import get_redis_pool

router = APIRouter()


def save_upload_file(upload_file: UploadFile) -> None:
    directory = uuid.uuid4()
    destination = Path(IMAGE_DIRECTORY) / str(directory) / 'original.jpg' # TODO file extension handling
    destination.parent.mkdir()
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return directory


@router.post("/detect/vehicles")
async def post_image(
    token: str = Body(...),
    image: UploadFile = File(...),
):
    pool = await get_redis_pool()

    key = save_upload_file(image)
    print(key)
    
    await pool.set(str(key), 'gotem')
    return dict(key=str(key))
        


@router.get("/detect/vehicles/{key}")
async def get_image(
    key: str,
):
    pool = await get_redis_pool()

    value = await pool.get(key)
    return dict(status=value)
