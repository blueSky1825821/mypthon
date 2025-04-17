"""
图像卷积
"""
from pytorch_test.convolutional_neural_network import *


def corr2d(X, K):  # @save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


class Conv2D(nn.Module):
    """
    卷积层：对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


"""
图像中目标的边缘检测
"""
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
# 当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零。
K = torch.tensor([[1.0, -1.0]])

Y = corr2d(X, K)
print(Y)
print(corr2d(X.t(), K))

"""
仅查看“输入-输出”对来学习由X生成Y的卷积核
"""
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    # 计算预测输出 Y_hat 和目标输出 Y 之间的均方误差（MSE）
    l = (Y_hat - Y) ** 2
    # 清除之前计算的梯度。这是因为在 PyTorch 中，梯度是累加的，
    # 如果不清除之前的梯度，新的梯度会与旧的梯度叠加，导致错误的梯度更新。
    conv2d.zero_grad()
    # 计算损失函数 l 的总和，并调用 backward() 方法进行反向传播。
    # 这一步会根据链式法则自动计算所有参数的梯度，包括卷积核的权重和偏置。
    l.sum().backward()
    # 迭代卷积核
    # 将权重减去学习率乘以梯度，从而实现梯度下降更新。
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))
