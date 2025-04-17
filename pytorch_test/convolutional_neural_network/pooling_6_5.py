from pytorch_test.convolutional_neural_network import *

"""
汇聚层：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性
"""


def pool2d(X, pool_size, mode='max'):
    """
    最大汇聚层（maximum pooling）和平均汇聚层（average pooling）:计算汇聚窗口中所有元素的最大值或平均值
    """
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))


#填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)

pool2d = nn.MaxPool2d(3)
print(pool2d(X))

#设定填充和步幅的高度和宽度
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))

#多个通道
X = torch.cat((X, X + 1), 1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))