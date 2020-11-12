# from raven import Client
from time import sleep
import redis

from celery import current_task

from app.config import CELERY_BACKEND_URI
from app.core.celery_app import celery_app
from app.core.db import get_redis_pool
from app.yolo_utils2 import detect_objects

# client_sentry = Client(settings.SENTRY_DSN)

print(CELERY_BACKEND_URI)
redis_store = redis.Redis.from_url(CELERY_BACKEND_URI)


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    redis_store.set('test', word)
    # for i in range(1, 2):
    #     sleep(1)
        #current_task.update_state(state='PROGRESS', meta={'process_percent': i*10})
    return f"test task return {word}"


@celery_app.task(acks_late=True)
def run_yolo(filename: str) -> None:
    pass

