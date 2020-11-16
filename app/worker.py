# from raven import Client
from pathlib import Path
import time

import cv2 as cv
from celery import current_task, Task

from app.config import IMAGE_DIRECTORY
from app.core.celery_app import celery_app
from app.yolo_utils2 import load_yolo_net, detect_objects, draw_detections, crop_detections
from app.wpod_utils import load_wpod_net, get_plate, draw_box, preprocess_image


from celery.signals import worker_process_init

wpod_net = None


@worker_process_init.connect()
def init_worker_process(**kwargs):
    global wpod_net
    wpod_net = load_wpod_net()


# client_sentry = Client(settings.SENTRY_DSN)

# redis_store = redis.Redis.from_url(CELERY_BACKEND_URI)


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    # redis_store.set('test', word)
    for i in range(1, 11):
        time.sleep(1)
        current_task.update_state(state='PROGRESS', meta={'process_percent': i*10})
    return f"test task return {word}"


class NNetTask(Task):
    """
    Base task class to load the neural nets when the work starts.
    """
    def __init__(self):
        yolo_net, yolo_labels, colors, layer_names = load_yolo_net()
        self.yolo_net = yolo_net
        self.yolo_labels = yolo_labels
        self.layer_names = layer_names
        self.colors = colors
        # wpod_net = load_wpod_net()
        # self.wpod_net = wpod_net


@celery_app.task(base=NNetTask, bind=True, acks_late=True)
def run_yolo(self, filename: str) -> None:
    filepath = Path(IMAGE_DIRECTORY) / self.request.id / filename
    current_task.update_state(state='PROGRESS', meta={'progress': 0.1})

    img = cv.imread(str(filepath))  # TODO move this into a helper function in yolo utils to hanle using cv
    detections = detect_objects(self.yolo_net, self.yolo_labels, self.layer_names, self.colors, img)

    detections_img = draw_detections(img.copy(), detections, self.colors, self.yolo_labels)
    save_path = filepath.parent / 'detections.jpg'
    cv.imwrite(str(save_path), detections_img)

    detection_images = crop_detections(img, detections)
    save_directory = filepath.parent / "objects" 
    if not save_directory.exists():
        save_directory.mkdir()
    for num, image in enumerate(detection_images):
        cv.imwrite(str(save_directory / f"{num+1}.jpg"), image)  # TODO fix cv2 error from some non-jpg


@celery_app.task(base=NNetTask, throws=(AssertionError), bind=True, acks_late=True)
def run_wpod(self, filename: str) -> None:
    filepath = Path(IMAGE_DIRECTORY) / self.request.id / filename
    current_task.update_state(state='PROGRESS', meta={'progress': 0.1})

    img = cv.imread(str(filepath))
    plateImg, cor = get_plate(wpod_net, img)  # XXX can raise AssertionError if no plate is found
    vehicleImg = draw_box(img, cor)
    cv.imwrite(str(filepath.parent / "plate.jpg"), plateImg[0] * 255)
    cv.imwrite(str(filepath.parent / "vehicle.jpg"), vehicleImg * 255)