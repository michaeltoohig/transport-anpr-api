import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Depends

from app.core.config import IMAGE_DIRECTORY
from app.worker import run_ocr, run_yolo, run_wpod
from app.api import deps

router = APIRouter()


@router.post("/detect/vehicles")
async def post_detect_image(
    request: Request,
    image: tuple = Depends(deps.upload_image),
) -> Any:
    taskId, filename = image
    task = run_yolo.apply_async(kwargs={"filename": filename}, task_id=taskId)
    return dict(taskId=task.id, statusUrl=request.url_for('detect-vehicles-results', taskId=task.id))


@router.get("/detect/vehicles/{taskId}", name="detect-vehicles-results")
async def get_detect_image(
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
            obj = {}
            url = request.url_for("images", path=f"{taskId}/objects/{file}")
            obj["src"] = url
            obj["file"] = file
            objs.append(obj)
        return dict(status=job.state, progress=1, image=image, objs=objs)


@router.post("/detect/plate")
async def post_detect_plate_image(
    request: Request,
    image: tuple = Depends(deps.vehicle_image),
    # token: str = Body(...),
) -> Any:
    taskId, filename = image
    task = run_wpod.apply_async(kwargs={"filename": filename}, task_id=taskId)
    return dict(taskId=task.id, statusUrl=request.url_for('detect-plate-results', taskId=task.id))


@router.get("/detect/plate/{taskId}", name="detect-plate-results")
async def get_detect_plate(
    request: Request,
    taskId: str,
    # token: str = Body(...),
) -> Any:
    job = run_wpod.AsyncResult(taskId)
    if job.state == 'PROGRESS':
        return dict(status=job.state, progress=job.result['progress'])
    elif job.state == 'SUCCESS':
        vehicle_image = request.url_for("images", path=f"{taskId}/vehicle.jpg")
        plate_image = request.url_for("images", path=f"{taskId}/plate.jpg")
        return dict(status=job.state, progress=1, vehicle=vehicle_image, plate=plate_image)
    else:
        return dict(status="FAILED", progress=1)


@router.post("/predict/plate")
async def post_predict_plate_image(
    request: Request,
    image: tuple = Depends(deps.plate_image),
    # token: str = Body(...),
) -> Any:
    taskId, filename = image
    task = run_ocr.apply_async(kwargs={"filename": filename}, task_id=taskId)
    return dict(taskId=task.id, statusUrl=request.url_for('predict-plate-results', taskId=task.id))


@router.get("/predict/plate/{taskId}", name="predict-plate-results")
async def get_predict_plate(
    request: Request,
    taskId: str,
    # token: str = Body(...),
) -> Any:
    job = run_ocr.AsyncResult(taskId)
    if job.state == 'PROGRESS':
        return dict(status=job.state, progress=job.result["progress"])
    elif job.state == 'SUCCESS':
        prediction = job.result
        return dict(status=job.state, progress=1, prediction=prediction)
    else:
        return dict(status='FAILED', progress=1)