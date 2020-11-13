# from raven import Client
from pathlib import Path
import time

import cv2 as cv
from celery import current_task, Task

from app.config import IMAGE_DIRECTORY
from app.core.celery_app import celery_app
from app.yolo_utils2 import load_yolo_net, detect_objects

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


@celery_app.task(base=NNetTask, bind=True, acks_late=True)
def run_yolo(self, filename: str) -> None:
    filepath = Path(IMAGE_DIRECTORY) / self.request.id / filename
    current_task.update_state(state='PROGRESS', meta={'state': 'Loading Image...'})

    img = cv.imread(str(filepath))  # TODO move this into a helper function in yolo utils to hanle using cv
    img, detections = detect_objects(self.yolo_net, self.yolo_labels, self.layer_names, self.colors, img)
    detectionsFilepath = filepath.parent / 'detections.jpg'
    cv.imwrite(str(detectionsFilepath), img)
