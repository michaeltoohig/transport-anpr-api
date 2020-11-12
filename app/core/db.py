import aioredis
from app.config import REDIS_HOST, REDIS_PORT, REDIS_DB_INDEX


async def get_redis_pool():
    try:
        pool = await aioredis.create_redis_pool(
            (REDIS_HOST, REDIS_PORT), db=REDIS_DB_INDEX, encoding='utf-8')
        return pool
    except ConnectionRefusedError as e:
        print('cannot connect to redis on:', REDIS_HOST, REDIS_PORT)
        return None
