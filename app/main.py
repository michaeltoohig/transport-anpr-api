import contextvars
from pathlib import Path

import aioredis
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from app.api.endpoints import router as api_router
from app.views.endpoints import router as web_router
from app.config import REDIS_HOST, REDIS_PORT, IMAGE_DIRECTORY
from app.yolo_utils2 import load_yolo_net
from app.ocr_utils import load_ocr_net
from app.wpod_utils import detect_plate, load_wpod_net


# cvar_redis = contextvars.ContextVar('redis', default=None)


# yolo_net, yolo_labels, layer_names = None, None, None
ocr_net, ocr_labels = None, None
wpod_net = None


app = FastAPI()
app.include_router(api_router, prefix="/api")
app.include_router(web_router, prefix="")
# Mount additional applications
app.mount("/static", StaticFiles(directory="app/views/static"), name="static")
app.mount("/images", StaticFiles(directory=IMAGE_DIRECTORY), name="images")

@app.on_event("startup")
async def handle_startup():
    if not Path(IMAGE_DIRECTORY).exists():
        Path(IMAGE_DIRECTORY).mkdir()
    
    # try:
    #     pool = await aioredis.create_redis_pool(
    #         (REDIS_HOST, REDIS_PORT), encoding='utf-8', maxsize=20)
    #     cvar_redis.set(pool)
    #     print("Connected to Redis on ", REDIS_HOST, REDIS_PORT)
    # except ConnectionRefusedError as e:
    #     print('cannot connect to redis on:', REDIS_HOST, REDIS_PORT)
    #     return

    load_yolo_net()
    # yolo_net, yolo_labels, layer_names = load_yolo_net()
    ocr_net, ocr_labels = load_ocr_net()
    wpod_net = load_wpod_net()


@app.on_event("shutdown")
async def handle_shutdown():
    # redis = cvar_redis.get()
    # redis.close()
    # await redis.wait_closed()
    print("closed connection Redis on ", REDIS_HOST, REDIS_PORT)
