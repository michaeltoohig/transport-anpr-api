import contextvars

import aioredis
from fastapi import FastAPI

from app.api.endpoints import router
from app.config import REDIS_HOST, REDIS_PORT
from app.yolo_utils2 import load_yolo_net
from app.ocr_utils import load_ocr_net
from app.wpod_utils import detect_plate, load_wpod_net


cvar_redis = contextvars.ContextVar('redis', default=None)


yolo_net = None
ocr_net, ocr_labels = None, None
wpod_net = None


app = FastAPI()
app.include_router(router)


@app.on_event("startup")
async def handle_startup():
    try:
        pool = await aioredis.create_redis_pool(
            (REDIS_HOST, REDIS_PORT), encoding='utf-8', maxsize=20)
        cvar_redis.set(pool)
        print("Connected to Redis on ", REDIS_HOST, REDIS_PORT)
    except ConnectionRefusedError as e:
        print('cannot connect to redis on:', REDIS_HOST, REDIS_PORT)
        return

    # yolo_net = load_yolo_net()
    # ocr_net, ocr_labels = load_ocr_net()
    # wpod_net = load_wpod_net()


@app.on_event("shutdown")
async def handle_shutdown():
    redis = cvar_redis.get()
    redis.close()
    await redis.wait_closed()
    print("closed connection Redis on ", REDIS_HOST, REDIS_PORT)
    