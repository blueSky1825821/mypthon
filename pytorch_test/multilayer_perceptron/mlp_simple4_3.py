from pytorch_test.multilayer_perceptron import *
from pytorch_test.linear_neural_network.softmax_regression3_6 import train_ch3, predict_ch3

net = nn.Sequential(nn.Flatten(),
                    # 输入层和隐藏层，隐藏层有256个神经元
                    nn.Linear(784, 256),
                    # ReLU激活函数
                    nn.ReLU(),
                    # 隐藏层和输出层，输出层有10个神经元
                    nn.Linear(256, 10))


def init_weights(m):
    # 初始化权重
    if type(m) == nn.Linear:
        # 正态分布初始化
        nn.init.normal_(m.weight, std=0.01)


# 初始化网络
net.apply(init_weights)
# 批量大小，学习率，迭代次数
batch_size, lr, num_epochs = 256, 0.1, 10
# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
