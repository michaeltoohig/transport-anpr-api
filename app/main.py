from typing import Optional

import uuid
import os
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel

from app.yolo_utils2 import load_yolo_net
from app.ocr_utils import load_ocr_net
from app.wpod_utils import detect_plate, load_wpod_net
from app.utils import 

yolo_net = None
ocr_net, ocr_labels = None, None
wpod_net = None

app = FastAPI()


@app.on_event("startup")
def startup():
    yolo_net = load_yolo_net()
    ocr_net, ocr_labels = load_ocr_net()
    wpod_net = load_wpod_net()
    

class PlatePrediction(BaseModel):
    predition: str


@app.post("/detect/vehicles")
def post_image(
    image: UploadFile = File(...),
    token: str = Form(...)
):
    filename = uuid.uuid4()
    img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    try:
        

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