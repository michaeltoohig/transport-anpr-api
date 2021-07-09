# from raven import Client
import json
from pathlib import Path

import colorgram
import cv2 as cv
import numpy as np
from PIL import Image
from celery import current_task, Task
from celery.signals import worker_process_init

from app.core.config import IMAGE_DIRECTORY
from app.core.celery_app import celery_app
from app.yolo_utils import VEHICLE_CLASSES, load_yolo_net, detect_objects, draw_detections, crop_detection
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


# TODO split into multiple sub tasks - run a task chain on API side
# one task for collecting detections from the image (which we can reuse in CLI)
# another for saving files and thumbs in directories
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

    # Get detections from image
    detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, filepath)
    current_task.update_state(state="PROGRESS", meta={"progress": 0.7})
    detections = list(filter(lambda d: d["label"] in VEHICLE_CLASSES, detections))

    # Save image with detections drawn 
    detections_img = draw_detections(filepath, detections)
    save_path = filepath.parent / "detections.jpg"
    cv.imwrite(str(save_path), detections_img)
    detections_img_sm = image_resize(detections_img)
    save_path = filepath.parent / "thumbs" / "detections.jpg"
    save_path.parent.mkdir()
    cv.imwrite(str(save_path), detections_img_sm)

    # Save image of cropped detections with JSON with the details of detection
    save_directory = filepath.parent / "objects" 
    if not save_directory.exists():
        save_directory.mkdir()
        (save_directory / "thumbs").mkdir()
    for num, obj in enumerate(detections):
        (save_directory / f"{num+1}.json").write_text(json.dumps(obj))
        image = crop_detection(filepath, obj)
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

    plateImg, cor = get_plate(wpod_net, filepath)  # XXX can raise AssertionError if no plate is found
    vehicleImg = draw_box(filepath, cor)
    
    # XXX Currently only handling one plate image from array of found images
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


@celery_app.task(
    base=BaseTask,
    bind=True,
    acks_late=True,
    soft_time_limit=10,
    time_limit=15,
)
def detect_colours(self, filename: str) -> None:
    MAX_WIDTH = 300
    filepath = Path(IMAGE_DIRECTORY) / self.request.id / filename
    current_task.update_state(state="PROGRESS", meta={"progress": 0.1})
    img = Image.open(filepath)
    if img.size[0] > MAX_WIDTH:
        width_ratio = MAX_WIDTH / float(img.size[0])
        new_height = int((float(img.size[1]) * float(width_ratio)))
        img = img.resize((MAX_WIDTH, new_height), Image.ANTIALIAS)
    current_task.update_state(state="PROGRESS", meta={"progress": 0.5})
    colours = colorgram.extract(img, 6)
    convertRGB = lambda c: "#{0:02x}{1:02x}{2:02x}".format(c.rgb.r, c.rgb.g, c.rgb.b).upper()
    return [dict(colour=convertRGB(c), proportion=c.proportion) for c in colours]