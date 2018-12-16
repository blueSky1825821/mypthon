import redis

# 线程池
pool = redis.ConnectionPool(host='47.96.154.180', port=6379, decode_responses=True)
conn = redis.Redis(connection_pool=pool)

print(conn.rpush('sort-input', 23, 15, 110, 7))
print(conn.sort('sort-input'))
print(conn.sort('sort-input', alpha=True))
print(conn.hset('d-7', 'field', 5))
print(conn.hset('d-15', 'field', 1))
print(conn.hset('d-23', 'field', 9))
print(conn.hset('d-110', 'field', 3))

print(conn.sort('sort-input', by='d-*->fiedl'))
print(conn.sort('sort-input', by='d-*->fiedl', get='d-*->field'))