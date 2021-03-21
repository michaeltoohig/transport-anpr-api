# from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from app.api.endpoints_v1 import router as api_router
from app.views.endpoints import router as web_router
from app.core.config import IMAGE_DIRECTORY

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://localhost:5000",
    "https://anpr.vehiclerank.vu",
    "https://vehiclerank.vu",
    "https://staging.vehiclerank.vu",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(web_router, prefix="")
app.mount("/static", StaticFiles(directory="app/views/static"), name="static")
app.mount("/images", StaticFiles(directory=IMAGE_DIRECTORY), name="images")


# @app.on_event("startup")
# async def handle_startup():
#     if not Path(IMAGE_DIRECTORY).exists():
#         Path(IMAGE_DIRECTORY).mkdir()
    
#     # try:
#     #     pool = await aioredis.create_redis_pool(
#     #         (REDIS_HOST, REDIS_PORT), encoding='utf-8', maxsize=20)
#     #     cvar_redis.set(pool)
#     #     print("Connected to Redis on ", REDIS_HOST, REDIS_PORT)
#     # except ConnectionRefusedError as e:
#     #     print('cannot connect to redis on:', REDIS_HOST, REDIS_PORT)
#     #     return

#     # load_yolo_net()
#     # # yolo_net, yolo_labels, layer_names = load_yolo_net()
#     # ocr_net, ocr_labels = load_ocr_net()
#     # wpod_net = load_wpod_net()


# @app.on_event("shutdown")
# async def handle_shutdown():
#     # redis = cvar_redis.get()
#     # redis.close()
#     # await redis.wait_closed()
#     print("closed connection Redis on ", REDIS_HOST, REDIS_PORT)
