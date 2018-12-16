# !/bin/python
listMilo = []
print(type(listMilo))
# 列表 可以改
listMilo = ['milo', 20, 'male']
# 元组 值不能被改，否则引用地址会变
t = ("milo", 30, "male")
print(listMilo[0])
print(t[0:2])
print(listMilo[0:2])

print(listMilo[1])
print(listMilo.remove(20))