#!/usr/bin/python3
def hello():
    print("Hello World!")


# 可写函数说明
def printinfo(name, age):
    "打印任何传入的字符串"
    print("名字: ", name)
    print("年龄: ", age)
    return


# 不定长参数
def printinfo2(arg1, *vartuple):
    "打印任何传入的字符串"
    print("输出： ")
    print(arg1)
    print(vartuple)


def sum(arg1: int, arg2: int) -> int:
    "返回两个参数的和."
    total = arg1 + arg2
    print("函数内： ", total)
    return total


# 匿名函数
sum2 = lambda arg1, arg2: arg1 + arg2


hello()
printinfo(age=50, name="runoob")
printinfo2(30, 2, 3)
sum(1, 2)
print("sum2: ", sum2(1, 2))
