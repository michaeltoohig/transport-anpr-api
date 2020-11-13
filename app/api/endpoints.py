from pathlib import Path
import shutil
from typing import Any
import uuid

from fastapi import APIRouter, Body, File, Request, UploadFile

from app.config import IMAGE_DIRECTORY
from app.worker import test_celery, run_yolo
from app.core.db import get_redis_pool

router = APIRouter()


def save_upload_file(upload_file: UploadFile) -> None:
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
        return dict(status=job.state, progress=job.result['current'] / job.result['total'])
    elif job.state == 'SUCCESS':
        image = request.url_for("images", path=f"{taskId}/detections.jpg")
        return dict(status=job.status, progress=1, image=image)


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