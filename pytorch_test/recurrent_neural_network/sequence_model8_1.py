"""
序列模型
自回归模型（autoregressive models）：假设在现实情况下相当长的序列 x_t-1,...,x_1可能是不必要的， 因此我们只需要满足某个长度为\tau 的时间跨度，
 即使用观测序列x_t-1,...,x_t-\tau。 当下获得的最直接的好处就是参数的数量总是不变的， 至少在t>\tau时如此，
 这就使我们能够训练一个上面提及的深度网络。对自己执行回归。
 隐变量自回归模型（latent autoregressive models）：保留一些对过去观测的总结h_t,并同时更新预测^x_t和总结h_t，
 ^x_t=P(x_t|h_t)估计x_t，以及公式h_t=g(h_t-1,x_t-1)更新的模型。h_t从未被观测到。参考隐变量自回归模型.svg

 整个序列的估计值公式：
 P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}, \ldots, x_1).

马尔可夫模型：使用x_{t-1}, \ldots, x_{t-\tau}而不是x_{t-1}, \ldots, x_1来估计x_t，只要这种是近似精确的，就说序列满足马尔可夫条件。

"""

import torch
from torch import nn
from d2l import torch as d2l

"""
使用正弦函数和一些可加性噪声来生成序列数据， 时间步为1,2,...,1000
"""
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

tau = 4
# features 的形状为 (996, 4)，表示从第 5 个时间步到第 1000 个时间步（共 996 个样本），每个样本包含过去 4 个时间步的特征。
features = torch.zeros((T - tau, tau))
"""
第一行：[x[0], x[1], x[2], x[3]]
第二行：[x[1], x[2], x[3], x[4]]
...
最后一行：[x[995], x[996], x[997], x[998]]
"""
for i in range(tau):
    features[:, i] = x[i: T - tau + i]

# 将结果 reshape 成一个列向量，形状为 (996, 1)。
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        # 旨在解决深度网络训练过程中梯度消失或爆炸的问题。
        # 它通过合理设置初始权重值，确保信号在前向传播和反向传播过程中不会迅速衰减或发散。
        nn.init.xavier_uniform_(m.weight)


# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net


# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
# 参数 reduction='none' 表示不对损失值进行平均或求和操作，而是返回每个样本的单独损失值。这通常用于需要对不同样本施加不同权重的情况。
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    # 使用 Adam 优化算法初始化优化器。
    # net.parameters() 表示需要优化的模型参数。
    # lr 表示学习率，用于控制优化过程中参数更新的幅度。
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            # 将模型参数的梯度清零，避免梯度累积
            trainer.zero_grad()
            # 计算模型的预测值 net(X) 与真实值 y 之间的损失。
            # 返回的 l 是一个张量，形状为 (batch_size, 1)，表示每个样本的损失值。
            l = loss(net(X), y)
            # 对损失张量求和后进行反向传播，计算梯度。
            l.sum().backward()
            # 根据计算出的梯度更新模型参数。
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)

"""
预测：
检查模型预测下一个时间步的能力，单步预测（one-step-ahead prediction）
"""
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         # [x.detach().numpy(), onestep_preds.detach().numpy()]：两个数据序列，分别表示原始数据和预测数据。
         # x.detach().numpy()：将原始数据 x 转换为 NumPy 数组。detach() 方法用于分离计算图，避免在绘图时影响梯度计算。
         [x.detach().numpy(), onestep_preds.detach().numpy()],
         # 分别表示 x 轴和 y 轴的标签。
         'time', 'x',
         # 分别表示原始数据和单步预测结果。
         legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

# 用于存储多步预测的结果
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        # 将提取的特征调整为模型输入所需的形状（1 行，tau 列）。
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))

"""
逐步预测：每次预测都基于之前 tau 步的预测结果，而不是真实值。
误差累积：随着预测步数的增加，误差可能会逐渐累积，导致预测结果偏离真实值。
"""
max_steps = 64

# 创建一个形状为 (T - tau - max_steps + 1, tau + max_steps) 的张量 features，并将其初始化为全零。
#列数为 tau + max_steps，前 tau 列为输入特征，后 max_steps 列为预测结果。
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    # 将输出调整为一维数组，并将其存储到 features 的第 i 列中。
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
