import torch
from torch import nn
from d2l import torch as d2l

"""
权重衰减（weight decay）是最广泛使用的正则化的技术之一，L2正则化
"""

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


def init_params():
    """
    初始化模型参数
    """
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    """
    定义L2范数惩罚：所有项求平方后并将它们求和。
    """
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    """
    将模型拟合训练数据集，并在测试数据集上进行评估
    """
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


train(lambd=0)

train(lambd=3)

"""
简洁实现
"""


def train_concise(wd):
    # 创建一个包含单个全连接层的神经网络 net，输入维度为 num_inputs，输出维度为 1。
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    # 使用正态分布初始化网络参数
    for param in net.parameters():
        param.data.normal_()
    # 均方误差损失函数 nn.MSELoss并设置 reduction='none' 表示不进行平均或求和操作
    loss = nn.MSELoss(reduction='none')
    # 置训练轮数 num_epochs 为 100，学习率 lr 为 0.003
    num_epochs, lr = 100, 0.003
    # 偏置参数不进行权重衰减，仅对权重参数应用权重衰减 wd。
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}], lr=lr)
    # 用于绘制训练和测试损失曲线
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    # 训练循环
    # 遍历每个epoch
    for epoch in range(num_epochs):
        # 遍历训练数据集中的每个批次
        for X, y in train_iter:
            # 清空梯度
            trainer.zero_grad()
            # 前向传播计算损失
            l = loss(net(X), y)
            # 反向传播计算梯度
            l.mean().backward()
            # 更新模型参数
            trainer.step()
            # 每5个epoch记录一次训练和测试损失，并更新绘图对象
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    # 打印权重参数的L2范
    print('w的L2范数：', net[0].weight.norm().item())


train_concise(0)
train_concise(3)
