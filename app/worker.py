# from raven import Client
from pathlib import Path

import cv2 as cv
import numpy as np
from celery import current_task, Task
from celery.signals import worker_process_init

from app.core.config import IMAGE_DIRECTORY
from app.core.celery_app import celery_app
from app.yolo_utils2 import VEHICLE_CLASSES, load_yolo_net, detect_objects, draw_detections, crop_detections
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
    # height, width = 
    save_path = filepath.parent / "detections.jpg"
    cv.imwrite(str(save_path), detections_img)

    detection_images = crop_detections(img, detections)
    save_directory = filepath.parent / "objects" 
    if not save_directory.exists():
        save_directory.mkdir()
    for num, image in enumerate(detection_images):
        cv.imwrite(str(save_directory / f"{num+1}.jpg"), image)  # TODO fix cv2 error from some non-jpg

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

    plateFile = str(filepath.parent / "plate.jpg")
    cv.imwrite(plateFile, plateImg * 255)
    cv.imwrite(str(filepath.parent / "vehicle.jpg"), vehicleImg * 255)

    if makePrediction:
        prediction = get_prediction(ocr_net, ocr_labels, plateFile)
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