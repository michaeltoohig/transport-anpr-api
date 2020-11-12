from app.config import CELERY_BROKER_URI, CELERY_BACKEND_URI
from celery import Celery

celery_app = Celery(
    "worker",
    include=['app.worker'],
    broker=CELERY_BROKER_URI, 
    backend=CELERY_BACKEND_URI
)

celery_app.conf.task_routes = {"app.worker.*": "main-queue"}