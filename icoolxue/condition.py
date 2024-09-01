# !/bin/python
if 1 < 2:
    print(1 < 2)


def fun():
    return 0


x = int(input("please input :"))

if x >= 90:
    print("A")
elif x < 60:
    print("B")
else:
    print("C")


print(1 and 1)
print(1 or 1)

var1 = 100
if var1:
    print("1 - if 表达式条件为 true")
    print(var1)

var2 = 0
if var2:
    print("2 - if 表达式条件为 true")
    print(var2)
print("Good bye!")


# while else
count = 0
while count < 5:
   print (count, " 小于 5")
   count = count + 1
else:
   print (count, " 大于或等于 5")

# 循环遍历
languages = ["C", "C++", "Perl", "Python"]
for x in languages:
    print(x)

# 内置range() 生成数列
for i in range(5):
    print(i)


for n in range(1, 6):
    if n == 1:
        print(n)
    else:
        print(n, "n != 1")

for letter in 'Runoob':
    if letter == 'o':
        # 不做任何事情，一般用做占位语句
        pass
        print('执行 pass 块')
    print('当前字母 :', letter)

print("Good bye!")