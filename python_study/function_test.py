def power(x, n=2):
    s = 1;
    while n > 0:
        n = n - 1
        s = s * x
    return s

print(power(5))
print(power(5, 2))

# 关键字参数：在函数内部自动组装成一个dict
def person(name, age, **kwargs):
    print('name: ', name, 'age: ', age, 'other: ', kwargs)

person('wang', '100')
person('wang', '100', city='Beijing')

extra = {'city':'Beijing', 'job':'Engineer'}
person('Jack', 24, city=extra['city'], job=extra['job'])
person('Jack', 24, **extra)

def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)
def f2(a, b, c=0, *, d, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)

f1(1, 2)
f1(1, 2, c=3)
f1(1, 2, 3, 'a', 'b')