from pytorch_test import *

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

X = x.reshape(3, 4)
print(X)

zeros = torch.zeros((2, 3, 4))
print(zeros)

tensor = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(tensor)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算

print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X, Y, torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

print(X == Y)

print(X.sum())

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)

print(X[-1], X[2:3])

print(X)
print(X[1, 2])

X[0:2, :] = 12
print(X[0:2, :])
print(X)

print(id(Y))
before = id(Y)
Y = Y + X
print(id(Y))
id(Y) == before

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

A = X.numpy()
B = torch.tensor(A)
print(id(X), type(A), A, id(A), type(B), B, id(B))

print("====")
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
