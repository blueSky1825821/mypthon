from pytorch_test.deep_learn import *

"""
参数管理：目标是找到使损失函数最小化的模型参数值
1、访问参数，用于调试、诊断和可视化；
2、参数初始化；
3、在不同模型组件间共享参数。
"""

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))
# 通过索引来访问模型的任意层,每层的参数都在其属性中
print(net[2].state_dict())

# 目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['2.bias'].data)


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)
print(rgnet[0][1][0].bias.data)


# 参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


# 初始化网络
net.apply(init_normal)
print([0].weight.data[0], net[0].bias.data[0])


# Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42。
def init_xavier(m):
    if type(m) == nn.Linear:
        # 旨在解决深度网络训练过程中梯度消失或爆炸的问题。
        # 它通过合理设置初始权重值，确保信号在前向传播和反向传播过程中不会迅速衰减或发散。
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        # 用均匀分布将线性层的权重初始化为范围在 ([-10, 10]) 内的随机值
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
net[0].weight[:2]

# 直接设置参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]

# 参数绑定
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
