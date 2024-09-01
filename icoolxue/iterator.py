# !/bin/python
import sys


class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        if self.a <= 20:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration


def fibonacci(n):  # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if counter > n:
            return
        yield a
        a, b = b, a + b
        counter += 1


f = fibonacci(10)  # f 是一个迭代器，由生成器返回生成

while True:
    try:
        print(next(f), end=" ")
    except StopIteration:
        sys.exit()


myclass = MyNumbers()
myiter = iter(myclass)

print(next(myclass))
print(next(myclass))
print(next(myclass))
print(next(myclass))
print(next(myclass))

list = [1, 2, 3, 4]
# 创建迭代器对象
it = iter(list)
# 输出迭代器的下一个元素
print(next(it))
print(next(it))

list = [1, 2, 3, 4]
# 创建迭代器对象
it = iter(list)
for x in it:
    print(x)

while True:
    try:
        print(next(it))
    except StopIteration:
        print("error")
        sys.exit()


