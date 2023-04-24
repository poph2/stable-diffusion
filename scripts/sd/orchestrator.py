import json
import os

import redis
from dotenv import load_dotenv

QUEUE_NAME = 'stable-diffusion-prompts'


class Orchestrator:

    def __init__(self) -> None:
        super().__init__()


if __name__ == '__main__':

    load_dotenv()

    r = redis.Redis(host=os.getenv('REDIS_HOST'),
                    port=int(os.getenv('REDIS_PORT')),
                    password=os.getenv('REDIS_PASSWORD'))

    count = 0

    while True:
        key, val = r.blpop(QUEUE_NAME)
        request = json.loads(val.decode("utf-8"))
        print(f'{count+1}: {val.decode("utf-8")}')
        count += 1
