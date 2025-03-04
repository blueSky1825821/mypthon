from pytorch_test.deep_learn import *

"""
生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层，
然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。
"""

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
# net.__call__(X)的简写
net(X)


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            # 在模块的参数初始化过程中， 系统知道在_modules字典中查找需要初始化参数的子块
            self._modules[str(idx)] = module

    def forward(self, X):
        """
        定义前向传播函数
        """
        for block in self._modules.values():
            # OrderedDict保证了按照成员添加的顺序遍历它们
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # requires_grad=False，表示这个张量不会参与梯度计算。这意味着在训练过程中，rand_weight 的值将保持不变。
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        # torch.mm 函数对 X 和 self.rand_weight 进行矩阵乘法，并加上偏置项 1。
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        # 由于这里复用了同一个层，实际上等价于两个全连接层共享了相同的参数。这种设计可以减少模型的参数数量，但也意味着这两个层的权重是完全相同的。
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
net(X)


# 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
