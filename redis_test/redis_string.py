import redis


# r = redis.Redis(host='47.96.154.180', port=6379, decode_responses=True)
# 线程池
pool = redis.ConnectionPool(host='47.96.154.180', port=6379, decode_responses=True)
conn = redis.Redis(connection_pool=pool)
conn.set('gender', 'male')
print(conn.get('gender'))
# 初始null
print(conn.get('key'))
# 默认0
print(conn.incr('key'))
print(conn.decr('key'))
print(conn.incr('key', 15))
print(conn.decr('key', 5))
print(conn.get('key'))
# 只要这个值可以被解释为整数，可以当作整数来处理
print(conn.set('key', '13'))
print(conn.incr('key'))

print("供Redis处理字串和二进制的命")
print(conn.append('new-string-key', 'hello '))
print(conn.append('new-string-key', 'world!'))
# 截取
print(conn.substr('new-string-key', 3, 7))
# 设定某个位置字符串
print(conn.setrange('new-string-key', 0, 'H'))
print(conn.get('new-string-key'))
print(conn.setrange('new-string-key', 11, ', how are you?'))
print(conn.setbit('another-key', 2, 1))
print(conn.setbit('another-key', 7, 1))
print(conn.get('another-key'))

print("Redis列表的推入操作和弹出操作")
print(conn.rpush('list-key', 'last'))
print(conn.lpush('list-key', 'first'))
print(conn.rpush('list-key', 'new last'))
print(conn.lrange('list-key', 0, -1))
print(conn.lpop('list-key'))
print(conn.lpop('list-key'))
print(conn.lrange('list-key', 0, -1))
print(conn.lpush('list-key', 'a', 'b', 'c'))
print(conn.lrange('list-key', 0, -1))
print(conn.ltrim('list-key', 2, -1))
print(conn.lrange('list-key', 0, -1))

print("阻塞式队列弹出命令及移动")
print(conn.rpush('list', 'item1'))
print(conn.rpush('list', 'item2'))
print(conn.rpush('list', 'item3'))
print(conn.brpoplpush('list2', 'list', 1))
print(conn.brpoplpush('list2', 'list', 1))
print(conn.lrange('list', 0, -1))
print(conn.brpoplpush('list', 'list2', 1))
print(conn.blpop(['list', 'list2'], 1))
print(conn.blpop(['list', 'list2'], 1))
print(conn.blpop(['list', 'list2'], 1))
print(conn.blpop(['list', 'list2'], 1))

print("集合命令")
print(conn.sadd('set-key', 'a', 'b', 'c'))
print(conn.srem('set-key', 'c', 'd'))
print(conn.srem('set-key', 'c', 'd'))
# 返回集合包含的元素数量
print(conn.scard('set-key'))
print(conn.smembers('set-key'))
# 将元素移动到另一个集合
print(conn.smove('set-key', 'set-key2', 'a'))
print(conn.smembers('set-key2'))

print("组合和处理多个集合的Redis")
print(conn.sadd('skey1', 'a', 'b', 'c', 'd'))
print(conn.sadd('skey2', 'c', 'd', 'e', 'f'))
print(conn.sdiff('skey1', 'skey2'))
print(conn.sinter('skey1', 'skey2'))
# 两个集合各不相同
print(conn.sunion('skey1', 'skey2'))

print("用于添加和删除键值对的散列操作")
# 添加多个键值对
print(conn.hmset('hash-key', {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}))
print(conn.hmget('hash-key', ['k2', 'k3']))
print(conn.hlen('hash-key'))
print(conn.hdel('hash-key', 'k1', 'k3'))

print("展示Redis散列的更高级特性")
print(conn.hmset('hash-key2', {'short': 'hello', 'long': 1000 * '1'}))
print(conn.hkeys('hash-key2'))
print(conn.hexists('hash-key2', 'num'))
print(conn.hincrby('hash-key2', 'num'))
print(conn.hexists('hash-key2', 'num'))

print("一些常用的有序集合命令")
# 成员，分值
print(conn.zadd('zset-key', 'a', 3, 'b', 2, 'c', 1))
print(conn.zcard('zset-key'))
# 自增
print(conn.zincrby('zset-key', 'c', 3))
# 获取得分
print(conn.zscore('zset-key', 'b'))
# 获取排名
print(conn.zrank('zset-key', 'c'))
# 统计
print(conn.zcount('zset-key', 0, 3))

