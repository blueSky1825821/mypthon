import logging
import time
import redis
import time
import threading

# 线程池
pool = redis.ConnectionPool(host='47.96.154.180', port=6379, decode_responses=True)
conn = redis.Redis(connection_pool=pool)

# 设置日志级别
SEVERITY = {
    logging.DEBUG: 'debug',
    logging.INFO: 'info',
    logging.WARNING: 'warning',
    logging.ERROR: 'error',
    logging.CRITICAL: 'critical',
}

SEVERITY.update((name, name) for name in SEVERITY.values())

def log_recent(conn, name, message, severity=logging.InFO, pipe=None):
    severity = str(SEVERITY.get(severity, severity)).lower()
    destination = 'recent: %s:%s'%(name, severity)
    message = time.asctime() + ' ' + message
    pipe = pipe or conn.pipeline()
    pipe.lpush(destination, message)
    pipe.ltrim(destination, 0, 99)
    pipe.execute()

