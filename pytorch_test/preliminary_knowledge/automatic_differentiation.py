import torch

from pytorch_test import *

x = torch.arange(4.0)
print(x)

"""
在我们计算关于的梯度(x处的导数)之前，需要一个地方来存储梯度。 
重要的是，我们不会在每次对一个参数求导时都分配新的内存。 
因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。 
注意，一个标量函数关于向量的梯度是向量，并且与具有相同的形状。
"""
# 则表示它可以参与求导，也可以从它向后求导。默认情况下，一个新的Variables 的 requires_grad 和 volatile 都等于 False 。
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
y = 2 * torch.dot(x, x)
print(y)

"""
通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
反向传播（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。
"""
y.backward()
print(x.grad)
print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
print(x.grad.zero_())
y = x.sum()
print(y)
y.backward()
print(x.grad)

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
print("x:", x)
y = x * x
print(y)
print(torch.ones(len(x)))
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
print(x.grad)

x.grad.zero_()
x = torch.arange(4.0)
x.requires_grad_(True)
print("x:", x)
y = x * x
print(y)
print(torch.ones(len(x)))
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print("x.grad", x.grad)


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
print("a", a)
d = f(a)
print("d", d)
print(d.backward())
print(a.grad == d / a)
