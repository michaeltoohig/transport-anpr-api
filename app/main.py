from typing import Optional

import uuid
import os
import cv2
import numpy as np
import contextvars

import aioredis
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel

from app.yolo_utils2 import load_yolo_net
from app.ocr_utils import load_ocr_net
from app.wpod_utils import detect_plate, load_wpod_net


REDIS_HOST = 'localhost'
REDIS_PORT = 6379
cvar_redis = contextvars.ContextVar('redis', default=None)


yolo_net = None
ocr_net, ocr_labels = None, None
wpod_net = None


async def get_redis_pool():
    try:
        pool = await aioredis.create_redis_pool(
            (REDIS_HOST, REDIS_PORT), encoding='utf-8')
        return pool
    except ConnectionRefusedError as e:
        print('cannot connect to redis on:', REDIS_HOST, REDIS_PORT)
        return None
    

app = FastAPI()


class PlatePrediction(BaseModel):
    predition: str


@app.get("/{value}")
async def get_value(
    value: str,
):
    pool = await get_redis_pool()
    print(pool)
    return {"got": value}


@app.post("/detect/vehicles")
async def post_image(
    image: UploadFile = File(...),
    token: str = Form(...)
):
    pool = await get_redis_pool()
    filename = uuid.uuid4()
    img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    try:
        pass
    except:
        pass
    return None
        

@app.post("/detect/plate")  #, response_model=PlatePrediction)
def post_vehicle_image(
    image: UploadFile = File(...), 
    token: str = Form(...)
):
    img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    try:
        prediction = get_prediction(img)
    except AssertionError:
        raise HTTPException(
            status_code=404,
            detail="Number plate not found in vehicle image"
        )
    if not prediction: # Check if there is at least one prediction
        raise HTTPException(
            status_code=404,
            detail="Number plate not found in vehicle image"
        )
    return {"prediction": prediction}


@app.post("/detect/vehicles")
def post_image():
    pass
    # Perhaps keep the detected vehicles in temp storage already cropped
    # return list of boxes detected in the response and give urls for each 
    # that can be called to get the detection of any plate in the image if exists.
    # after awhile they will be deleted.



@app.on_event("startup")
async def handle_startup():
    try:
        pool = await aioredis.create_redis_pool(
            (REDIS_HOST, REDIS_PORT), encoding='utf-8', maxsize=20)
        cvar_redis.set(pool)
        print("Connected to Redis on ", REDIS_HOST, REDIS_PORT)
    except ConnectionRefusedError as e:
        print('cannot connect to redis on:', REDIS_HOST, REDIS_PORT)
        return

    yolo_net = load_yolo_net()
    ocr_net, ocr_labels = load_ocr_net()
    wpod_net = load_wpod_net()


@app.on_event("shutdown")
async def handle_shutdown():
    redis = cvar_redis.get()
    redis.close()
    await redis.wait_closed()
    print("closed connection Redis on ", REDIS_HOST, REDIS_PORT)
    