#!/bin/python
f = open("file.md", "w+")
f.write("Python 是一个非常好的语言。\n是的，的确非常好!!\n是的，的确非常好!!\n")
f.close()

r = open("file.md", "r")
str = r.readline()
print(str)
str = r.readlines()
print(str)
r.close()

r = open("file.md", "r")
for line in r:
    print(line, end='')
r.close()
