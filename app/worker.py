# from raven import Client
from time import sleep

from celery import current_task

from app.core.celery_app import celery_app
from app.yolo_utils2 import detect_objects

# client_sentry = Client(settings.SENTRY_DSN)


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    for i in range(1, 11):
        sleep(1)
        current_task.update_state(state='PROGRESS', meta={'process_percent': i*10})
    return f"test task return {word}"


@celery_app.task(acks_late=True)
def run_yolo(filename: str) -> None:
    pass

