# set
basket = {'apple', "orange", 'apple'}
print(basket)

print('orange' in basket)

# 集合的运算
a = set('abcdef')
b = set('abced')
print(a)
print(b)
# 集合a中包含而集合b中不包含的元素
print(a - b)
# 集合a或b中包含的所有元素
print(a | b)
# 不同时包含于a和b的元素
print(a ^ b)
# 集合a和b中都包含了的元素
print(a & b)

thisset = set(("Google", "Runoob", "Taobao"))
thisset.add("Facebook")
print(thisset)
thisset.update({'Alibaba'})
thisset.update([1, 2, 4])
print(thisset)
# 不存在不会发生错误
thisset.discard(3)
print(thisset)
thisset.discard(2)
print(thisset)
# 不存在会发生错误
# thisset.remove(3)
print(thisset)

x = thisset.pop()
print(x)

len(thisset)

print(x in thisset)
thisset.clear()
print(thisset)