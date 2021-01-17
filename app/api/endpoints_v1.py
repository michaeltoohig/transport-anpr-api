import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Depends, Query

from app.core.config import IMAGE_DIRECTORY
from app.worker import MockColorgramColour, detect_colours, run_ocr, run_yolo, run_wpod
from app.api import deps

router = APIRouter()


@router.post("/detect/vehicles", status_code=201)
async def post_detect_image(
    request: Request,
    image: tuple = Depends(deps.upload_image),
    # token: str = Query(...),
) -> Any:
    taskId, filename = image
    task = run_yolo.apply_async(kwargs={"filename": filename}, task_id=taskId)
    return dict(taskId=task.id, statusUrl=request.url_for('detect-vehicles-results', taskId=task.id))


@router.get("/detect/vehicles/{taskId}", name="detect-vehicles-results")
async def get_detect_image(
    request: Request,
    taskId: str,
    # token: str = Query(...),
) -> Any:
    job = run_yolo.AsyncResult(taskId)
    if job.state == 'PENDING':
        return dict(status=job.state, progress=0)
    elif job.state == 'FAILURE':
        return dict(status=job.state, progress=0)
    elif job.state == 'PROGRESS':
        return dict(status=job.state, progress=job.result['progress'])
    elif job.state == 'SUCCESS':
        detections_thumb = request.url_for("images", path=f"{taskId}/thumbs/detections.jpg")
        detected_objs = []
        for file in os.listdir(str(Path(IMAGE_DIRECTORY) / taskId / "objects" / "thumbs")):
            obj = {}
            url = request.url_for("images", path=f"{taskId}/objects/thumbs/{file}")
            obj["src"] = url
            obj["file"] = file
            detected_objs.append(obj)
        return dict(status=job.state, progress=1, image=detections_thumb, objs=detected_objs, taskId=taskId, detectPlateUrl=request.url_for("detect-plate"))


@router.post("/detect/plate", name="detect-plate", status_code=201)
async def post_detect_plate_image(
    request: Request,
    image: tuple = Depends(deps.vehicle_image),
    makePrediction: bool = Query(False),
    # token: str = Query(...),
) -> Any:
    taskId, filename = image
    task = run_wpod.apply_async(kwargs={"filename": filename, "makePrediction": makePrediction}, task_id=taskId)
    return dict(taskId=task.id, statusUrl=request.url_for('detect-plate-results', taskId=task.id))


@router.get("/detect/plate/{taskId}", name="detect-plate-results")
async def get_detect_plate(
    request: Request,
    taskId: str,
    # token: str = Query(...),
) -> Any:
    job = run_wpod.AsyncResult(taskId)
    if job.state == 'PENDING':
        return dict(status=job.state, progress=0)
    elif job.state == 'FAILURE':
        return dict(status=job.state, progress=0)
    elif job.state == 'PROGRESS':
        return dict(status=job.state, progress=job.result['progress'])
    elif job.state == 'SUCCESS':
        vehicle_image = request.url_for("images", path=f"{taskId}/thumbs/vehicle.jpg")
        plate_image = request.url_for("images", path=f"{taskId}/plate.jpg")
        prediction = job.result if job.result else ""
        return dict(status=job.state, progress=1, vehicle=vehicle_image, plate=plate_image, prediction=prediction)
    else:
        return dict(status="FAILURE", progress=1)


@router.post("/predict/plate", status_code=201)
async def post_predict_plate_image(
    request: Request,
    image: tuple = Depends(deps.plate_image),
    # token: str = Query(...),
) -> Any:
    taskId, filename = image
    task = run_ocr.apply_async(kwargs={"filename": filename}, task_id=taskId)
    return dict(taskId=task.id, statusUrl=request.url_for('predict-plate-results', taskId=task.id))


@router.get("/predict/plate/{taskId}", name="predict-plate-results")
async def get_predict_plate(
    request: Request,
    taskId: str,
    # token: str = Query(...),
) -> Any:
    job = run_ocr.AsyncResult(taskId)
    if job.state == 'PENDING':
        return dict(status=job.state, progress=0)
    elif job.state == 'FAILURE':
        return dict(status=job.state, progress=0)
    if job.state == 'PROGRESS':
        return dict(status=job.state, progress=job.result["progress"])
    elif job.state == 'SUCCESS':
        prediction = job.result
        return dict(status=job.state, progress=1, prediction=prediction)
    else:
        return dict(status='FAILURE', progress=1)


@router.post("/detect/colours", status_code=201)
async def post_detect_colours(
    request: Request,
    image: tuple = Depends(deps.vehicle_image),
    # token: str = Query(...),
) -> Any:
    taskId, filename = image
    task = detect_colours.apply_async(kwargs={"filename": filename}, task_id=taskId)
    return dict(taskId=task.id, statusUrl=request.url_for('detect-colours-results', taskId=task.id))


@router.get("/detect/colours/{taskId}", name="detect-colours-results")
async def get_detect_colours(
    request: Request,
    taskId: str,
    # token: str = Query(...),
) -> Any:
    colours = []
    for _ in range(6):
        colours.append(MockColorgramColour())
    return [dict(
        rgb=c.rgb,
        proportion=c.proportion,
    ) for c in colours]

    # job = detect_colours.AsyncResult(taskId)
    # if job.state == 'SUCCESS':
    #     import pdb; pdb.set_trace()
    #     pass
    # else:
    #     return dict(status=job.state, progress=0)


    # if job.state == 'PENDING':
    #     return dict(status=job.state, progress=0)
    # elif job.state == 'FAILURE':
    #     return dict(status=job.state, progress=0)
    # elif job.state == 'PROGRESS':
    #     return dict(status=job.state, progress=job.result['progress'])
    # elif job.state == 'SUCCESS':
    #     detections_thumb = request.url_for("images", path=f"{taskId}/thumbs/detections.jpg")
    #     detected_objs = []
    #     for file in os.listdir(str(Path(IMAGE_DIRECTORY) / taskId / "objects" / "thumbs")):
    #         obj = {}
    #         url = request.url_for("images", path=f"{taskId}/objects/thumbs/{file}")
    #         obj["src"] = url
    #         obj["file"] = file
    #         detected_objs.append(obj)
    #     return dict(status=job.state, progress=1, image=detections_thumb, objs=detected_objs, taskId=taskId, detectPlateUrl=request.url_for("detect-plate"))
