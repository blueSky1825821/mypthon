import torch
from d2l import torch as d2l

# 梯度消失
# 创建一个从-8到8的张量x，步长为0.1，并设置requires_grad=True以追踪梯度。
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
# 反向传播 torch.ones_like(x) 创建一个与 x 形状相同的全1张量，作为反向传播的初始梯度值。
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
d2l.plt.show()

# 梯度爆炸
#生成100个高斯随机矩阵，并将它们与某个初始矩阵相乘。
#矩阵乘积发生爆炸，是由于深度网络的初始化所导致时，我们没有机会让梯度下降优化器收敛。
M = torch.normal(0, 1, size=(4, 4))
print('一个矩阵 \n', M)
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)
