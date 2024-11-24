from pytorch_test.linear_neural_network import *

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

"""定义模型"""
"""第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1"""
"""全连接层"""
net = nn.Sequential(nn.Linear(2, 1))

"""初始化模型参数"""
"""指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零。"""
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

"""定义损失函数"""
"""计算均方误差使用的是MSELoss类，也称为平方范数。 默认情况下，它返回所有样本损失的平均值。"""
loss = nn.MSELoss()

"""定义优化算法"""
""" 当我们实例化一个SGD实例时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。 
小批量随机梯度下降只需要设置lr值，这里设置为0.03。"""
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""训练"""
"""
通过调用net(X)生成预测并计算损失l（前向传播）。
通过进行反向传播来计算梯度。
通过调用优化器来更新模型参数。
"""
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
