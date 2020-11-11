# from raven import Client

from app.core.celery_app import celery_app
from app.yolo_utils2 import detect_objects

# client_sentry = Client(settings.SENTRY_DSN)



@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    return f"test task return {word}"

    
@celery_app.task(acks_late=True)
def run_yolo(filename: str) -> None:
    pass

