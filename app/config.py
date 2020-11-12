import os

IMAGE_DIRECTORY = "./_images"

REDIS_HOST = os.environ.get("REDIS_HOST", 'localhost')
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
REDIS_DB_INDEX = os.environ.get("REDIS_DB_INDEX", 1)

CELERY_BROKER_URI = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_INDEX}"
CELERY_BACKEND_URI = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_INDEX}"