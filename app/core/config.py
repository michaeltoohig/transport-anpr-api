import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

SERVER_HOST = os.environ.get("SERVER_HOST", "http://localhost:8000")
IMAGE_DIRECTORY = os.environ.get("IMAGE_DIRECTORY", "./_images")

CORS_ORIGINS = os.environ.get("CORS_ORIGINS").split(",") if os.environ.get("CORS_ORIGINS") else None

REDIS_HOST = os.environ.get("REDIS_HOST", 'localhost')
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_DB_INDEX = os.environ.get("REDIS_DB_INDEX", 1)

CELERY_BROKER_URI = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_INDEX}"
CELERY_BACKEND_URI = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_INDEX}"
