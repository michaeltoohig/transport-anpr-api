import uuid
import shutil
from typing import Union
from pathlib import Path

from fastapi import UploadFile, File, Depends, HTTPException

from app.core.config import IMAGE_DIRECTORY


async def upload_image(image: UploadFile = File(None)):
    if not image:
        return None
    directory = uuid.uuid4()
    filename = 'original.jpg'  # TODO file extension handling
    destination = Path(IMAGE_DIRECTORY) / str(directory) / filename
    destination.parent.mkdir()
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    finally:
        image.file.close()
    return str(directory), filename


async def detected_vehicle(taskId: str = None, file: str = None):
    if not taskId and not file:
        return None
    vehicleImg = Path(IMAGE_DIRECTORY) / taskId / "objects" / file
    if not vehicleImg.exists():
        return None
    directory = uuid.uuid4()
    filename = "original.jpg"  # TODO file extension handling
    destination = Path(IMAGE_DIRECTORY) / str(directory) / filename
    destination.parent.mkdir()
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(vehicleImg.open("rb"), buffer)
    finally:
        pass
    return str(directory), filename


async def vehicle_image(
    new_image: Union[tuple, None] = Depends(upload_image), 
    old_image: Union[tuple, None] = Depends(detected_vehicle),
) -> tuple:
    if new_image is None and old_image is None:
        raise HTTPException(
            status_code=400,
            detail="No image given"
        )
    return new_image or old_image