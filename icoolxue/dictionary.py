# !/bin/python
# 字典 是python中唯一的映射类型 map
t = ['name', 'age', 'gender']
t2 = ['milo', 30, 'male']

print(zip(t, t2))

dic = {0: 0, 1: 1, 2: 2}
print(dic[0])
print(dic[1])

for k in dic:
    print(dic[k])

dic[3] = 3

for k in dic:
    print(dic[k])

dic.pop(3)

for k in dic:
    print(dic[k])

print(dict.items(dic))

print(dic.keys())

del dic

# 删除
for k in dic:
    print(dic[k])

