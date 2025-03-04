from pytorch_test.multilayer_perceptron import *
from pytorch_test.linear_neural_network.softmax_regression3_6 import train_ch3, predict_ch3

"""
多层感知机从零实现
"""

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""
初始化模型参数
Fashion-MNIST中的每个图像由28*28=784个灰度像素值组成。，所有图像共分为10个类别。
忽略像素之间的空间结构，每个图像视为784个输入特征和10个类的简单分类数据集。
实现一个具有单隐藏层的多层感知机，含256个隐藏单元
每一层都要记录一个权重矩阵和一个偏置向量，为损失关于这些参数的梯度分配内存
"""
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

"""
激活函数（activation function）通过计算加权并加上偏置来确定神经元是否应该被激活，将输入信号转换为输出的可微运算。大多数都是非线性。
ReLU激活函数：修正线性单元（Rectified linear unit，ReLU），ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。 
"""


def relu(X):
    # 创建一个与输入张量 X 形状相同、元素全为0的张量 a。
    a = torch.zeros_like(X)
    # 返回一个新的张量，其每个元素是 X 和 a 对应位置元素的最大值。
    # 这样实现了ReLU函数的效果，即所有负数被置为0，正数保持不变。
    return torch.max(X, a)


"""
模型
忽略了空间结构，使用reshape将每个二维图像转换为一个长度为num_inputs的向量
"""


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return (H @ W2 + b2)


"""
损失函数
计算交叉熵损失
"""
loss = nn.CrossEntropyLoss(reduction='none')

"""
训练
同softmax回归的训练过程
"""
# 将迭代周期数设置为10，并将学习率设置为0.1
num_epochs, lr = 10, 0.1
# 初始化一个随机梯度下降（SGD）优化器，用于更新模型参数
# params 是需要优化的模型参数
# lr 是学习率，控制参数更新的步长
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
predict_ch3(net, test_iter)
