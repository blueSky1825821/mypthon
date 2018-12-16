import redis
import time
import threading

# 线程池
pool = redis.ConnectionPool(host='47.96.154.180', port=6379, decode_responses=True)
conn = redis.Redis(connection_pool=pool)


def notrans():
    print(conn.incr('notrans:'))
    time.sleep(.1)
    conn.incr('notrans:', -1)


if 1:
    for i in range(3):
        threading.Thread(target=notrans()).start()
    time.sleep(.5)
