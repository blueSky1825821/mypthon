#!/usr/bin/python3
from collections import deque

a = [66.25, 333, 333, 1, 1234.5]
print(a.count(333), a.count(1), a.count("x"))
print(a.insert(2, -1))
print(a.append(333))
print(a.index(333))
a.reverse()
print(a)
a.sort()
print(a)
a.pop()
print(a)
a.pop()
print(a)

queue = deque(a)
queue.append(21)
queue.append(211)
print(queue)
queue.popleft()
print(queue)
