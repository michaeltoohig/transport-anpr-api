import os
from pathlib import Path
import shutil
from typing import Any, Tuple
import uuid

from fastapi import APIRouter, Body, File, Request, UploadFile

from app.config import IMAGE_DIRECTORY
from app.worker import test_celery, run_yolo
from app.core.db import get_redis_pool

router = APIRouter()


def save_upload_file(upload_file: UploadFile) -> Tuple[str, str]:
    directory = uuid.uuid4()
    filename = 'original.jpg' # TODO file extension handling
    destination = Path(IMAGE_DIRECTORY) / str(directory) / filename
    destination.parent.mkdir()
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return str(directory), filename

def get_upload_file_detection(taskId: str, objNum: int) -> None:
    file = Path(IMAGE_DIRECTORY) / taskId / "objects" / f"{objNum}.jpg"
    if not file.exists():
        raise Exception
    return str(file)


@router.post("/detect")
async def post_detect_image(
    request: Request,
    image: UploadFile = File(...),
) -> Any:
    #pool = await get_redis_pool()
    #await pool.set(str(key), 'gotem')
    
    taskId, filename = save_upload_file(image)
    task = run_yolo.apply_async(kwargs={"filename": filename}, task_id=taskId)

    return dict(taskId=task.id, statusUrl=request.url_for('detect-results', taskId=task.id))
        

@router.get("/detect/{taskId}", name="detect-results")
async def get_detect(
    request: Request,
    taskId: str,
    # token: str = Body(...),
) -> Any:
    job = run_yolo.AsyncResult(taskId)
    if job.state == 'PROGRESS':
        return dict(status=job.state, progress=job.result['progress'])
    elif job.state == 'SUCCESS':
        image = request.url_for("images", path=f"{taskId}/detections.jpg")
        objs = []
        for file in os.listdir(str(Path(IMAGE_DIRECTORY) / taskId / "objects")):
            print(file)
            url = request.url_for("images", path=f"{taskId}/objects/{file}")
            objs.append(url)
        return dict(status=job.status, progress=1, image=image, objs=objs)


@router.post("/detect/{taskId}/{objNum}/plate")
async def post_detect_plate(
    request: Request,
    taskId: str,
    objNum: int,
    # token: str = Body(...),
) -> Any:
    file = get_upload_file_detection(taskId, objNum)
    task = run_


@router.get("/test-celery/{word}")
async def test_celery_task(word: str):
    test_celery.delay(word=word)
    return True


@router.get("/test-celery")
async def status():
    pool = await get_redis_pool()
    resp = await pool.get('test')
    print(resp)
    return {"answer": resp}