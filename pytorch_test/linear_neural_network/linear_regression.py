from pytorch_test import *
from pytorch_test.linear_neural_network.my_timer import Timer

"""
线性回归
@link:https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression.html

回归（regression）：为一个或多个自变量与因变量之间关系建模的一类方法。机器学习中大多数任务通常都与预测prediction相关

example： 我们希望根据房屋的面积（平方英尺）和房龄（年）来估算房屋价格（美元）。
数据集包括房屋的销售价格、面积和房龄
机器学习中，该数据集称为训练数据集(training data set)或训练集(training set)
样本：每行数据（一次交易对应的数据）
标签label或目标target：试图预测的目标（预测房屋价格）
特征（协变量）：预测所依据的自变量（面积和房龄）

损失函数：能够量化目标的实际值与预测值之间的差距
解析解：线性回归的解能用公式简单的表达，不是所有问题都存在解析解

随机梯度下降：计算损失函数（数据集中所有样本的损失均值）关于模型参数的导数（梯度）。
实际中可能非常慢：每一次更新参数之前，必须便利整个数据集。
小批量随机梯度下降：通常会在每次需要计算更新的时候随机抽取一小批样本。

超参数（hyperparameter）：可以调整但不在训练过程中更新的参数。
调参（hyperparameter）是选择超参数的过程
超参通常是根据训练迭代结果来调整，训练迭代结果是在独立的验证数据集上评估得到的

泛化（generalization）：找到一组参数，能够在我们从见过的数据上实现较低的损失

预测（prediction）或推断（inference）：戈丁特征估计目标的过程

神经网络图
线性回归模型可视为单个神经元组成的神经网络，或称为单层神经网络
全连接层（full-connected layer）或稠密层（dense layer）：对于线性回归，每个输入都与每个输出相连
"""

n = 10000
a = torch.ones([n])
b = torch.ones([n])

c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'

# 矢量化代码通常会带来数量级的加速
timer.start()
d = a + b
f'{timer.stop():.5f} sec'


# 正态分布
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


# 再次使用numpy进行可视化
x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show()
