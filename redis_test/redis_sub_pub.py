import redis
import time
import threading

# 线程池
pool = redis.ConnectionPool(host='47.96.154.180', port=6379, decode_responses=True)
conn = redis.Redis(connection_pool=pool)


def publisher(n):
    time.sleep(1)
    for i in range(n):
        conn.publish('channle', i)
        time.sleep(1)


def run_pubsub():
    # 启动发送者线程，发送三条消息
    threading.Thread(target=publisher, args=(3,)).start()
    # 创建发布订阅对象
    pubsub = conn.pubsub()
    pubsub.subscribe(['channel'])
    count = 0
    # 遍历 监听订阅消息
    for item in pubsub.listen():
        print(item)
        count += 1
        if count == 4:
            # 取消订阅
            pubsub.unsubscribe()
        if count == 5:
            break


if __name__ == "__main__":
    run_pubsub()
