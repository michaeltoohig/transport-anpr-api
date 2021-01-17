# from raven import Client
import json
import random
from pathlib import Path
from typing import Tuple

import cv2 as cv
import numpy as np
from celery import current_task, Task
from celery.signals import worker_process_init

from app.core.config import IMAGE_DIRECTORY
from app.core.celery_app import celery_app
from app.yolo_utils2 import VEHICLE_CLASSES, load_yolo_net, detect_objects, draw_detections, crop_detection
from app.wpod_utils import load_wpod_net, get_plate, draw_box
from app.ocr_utils import load_ocr_net, get_prediction

# client_sentry = Client(settings.SENTRY_DSN)

yolo_net, yolo_labels, yolo_colors, yolo_layers = None, None, None, None
wpod_net = None
ocr_net, ocr_labels = None, None


@worker_process_init.connect()
def init_worker_process(**kwargs):
    global yolo_net, yolo_labels, yolo_colors, yolo_layers
    yolo_net, yolo_labels, yolo_colors, yolo_layers = load_yolo_net()
    global wpod_net
    wpod_net = load_wpod_net()
    global ocr_net, ocr_labels
    ocr_net, ocr_labels = load_ocr_net()


class BaseTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log the exceptions."""
        # client_sentry
        super(BaseTask, self).on_failure(exc, task_id, args, kwargs, einfo)


def image_resize(img):
    maxWidth = 300
    height, width = img.shape[:2]
    if width > maxWidth:
        height = int(height * maxWidth / width)
        width = maxWidth
    return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)


@celery_app.task(
    base=BaseTask,
    bind=True, 
    acks_late=True,
    soft_time_limit=10,
    time_limit=15,
)
def run_yolo(self, filename: str) -> None:
    filepath = Path(IMAGE_DIRECTORY) / self.request.id / filename
    current_task.update_state(state="PROGRESS", meta={"progress": 0.1})

    img = cv.imread(str(filepath))  # TODO move this into a helper function in yolo utils to handle using cv
    detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, img)
    current_task.update_state(state="PROGRESS", meta={"progress": 0.7})

    detections = list(filter(lambda d: d["label"] in VEHICLE_CLASSES, detections))

    detections_img = draw_detections(img.copy(), detections)
    save_path = filepath.parent / "detections.jpg"
    cv.imwrite(str(save_path), detections_img)
    detections_img_sm = image_resize(detections_img)
    save_path = filepath.parent / "thumbs" / "detections.jpg"
    save_path.parent.mkdir()
    cv.imwrite(str(save_path), detections_img_sm)

    save_directory = filepath.parent / "objects" 
    if not save_directory.exists():
        save_directory.mkdir()
        (save_directory / "thumbs").mkdir()
    for num, obj in enumerate(detections):
        # Save JSON file with bounding box to original photo
        (save_directory / f"{num+1}.json").write_text(json.dumps(obj))
        # Save cropped image of detected object and thumbnail
        image = crop_detection(img, obj)
        cv.imwrite(str(save_directory / f"{num+1}.jpg"), image)  # TODO fix cv2 error from some non-jpg
        image_sm = image_resize(image)
        cv.imwrite(str(save_directory / "thumbs" / f"{num+1}.jpg"), image_sm)

    return detections


@celery_app.task(
    base=BaseTask,
    bind=True,
    acks_late=True,
    soft_time_limmit=10,
    time_limit=15,
    throws=(AssertionError),
)
def run_wpod(self, filename: str, makePrediction: bool = False) -> None:
    filepath = Path(IMAGE_DIRECTORY) / self.request.id / filename
    current_task.update_state(state="PROGRESS", meta={"progress": 0.1})

    img = cv.imread(str(filepath))
    plateImg, cor = get_plate(wpod_net, img)  # XXX can raise AssertionError if no plate is found
    vehicleImg = draw_box(img, cor)
    
    img_float32 = np.float32(plateImg[0])
    plateImg = cv.cvtColor(img_float32, cv.COLOR_RGB2BGR)
    
    img_float32 = np.float32(vehicleImg)
    vehicleImg = cv.cvtColor(img_float32, cv.COLOR_RGB2BGR)

    plateFile = filepath.parent / "plate.jpg"
    cv.imwrite(str(plateFile), plateImg * 255)
    cv.imwrite(str(filepath.parent / "vehicle.jpg"), vehicleImg * 255)

    # plateImgSm = image_resize(plateImg)
    # cv.imwrite(str(plateFile.parent / "thumbs" / "plate.jpg"), plateImgSm * 255)
    vehicleImgSm = image_resize(vehicleImg)
    (filepath.parent / "thumbs").mkdir()
    cv.imwrite(str(filepath.parent / "thumbs" / "vehicle.jpg"), vehicleImgSm * 255)
    
    if makePrediction:
        prediction = get_prediction(ocr_net, ocr_labels, str(plateFile))
        return prediction


@celery_app.task(
    base=BaseTask,
    bind=True, 
    acks_late=True,
    soft_time_limit=5,
    time_limit=10,
)
def run_ocr(self, filename: str) -> None:
    filepath = Path(IMAGE_DIRECTORY) / self.request.id / filename
    current_task.update_state(state="PROGRESS", meta={"progress": 0.1})

    prediction = get_prediction(ocr_net, ocr_labels, str(filepath))
    return prediction


class MockColorgramColour(object):
    rgb: Tuple = None
    proportion: float = None

    def __init__(self):
        self.rgb = self._generate_random_color()
        self.proportion = random.random()

    def __repr__(self):
        return json.dumps(dict(rgb=self.rgb, proportion=self.proportion))

    def _generate_random_color(self):
        a = int(random.random() * 255)
        b = int(random.random() * 255)
        c = int(random.random() * 255)
        return (a, b, c)


@celery_app.task(
    base=BaseTask,
    bind=True,
    acks_late=True,
    soft_time_limit=5,
    time_limit=10,
)
def detect_colours(self, filename: str) -> None:
    colours = []
    for i in range(3):
        colours.append(MockColorgramColour())
    print(colours)