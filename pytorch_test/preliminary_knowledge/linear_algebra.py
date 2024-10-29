import torch

from pytorch_test import *

"""
最大化分配给观测数据的概率; 最小化预测和真实观测之间的距离。 
用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。 
目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。
"""


def scalar_operation():
    print("标量操作")
    # 标量操作 标量由只有一个元素的张量表示
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    print(x + y, x * y, x / y, x ** y)


def vector_operation():
    print("向量操作")
    # 向量可以被视为标量值组成的列表
    # 标量值被称为向量的元素（element）或分量（component）
    # 粗体 小写 x y z 表示向量
    x = torch.arange(4)
    print("x:", x)
    print("张量索引访问任一元素x[2]:", x[2])
    print("张量长度:", len(x))
    print("只有一个轴的张量(即是向量)的长度:", x.shape)


def matrix_operation():
    print("矩阵操作")
    # 向量将标量从零阶推广到一阶，矩阵(大写字母表示)将向量从一阶推广到二阶
    A = torch.arange(20).reshape(5, 4)
    print("矩阵A:", A)
    # 通过行索引（i）和列索引（j）来访问矩阵中的标量元素aij， 例如[A]ij。
    # 矩阵转置
    print("矩阵A转置:", A.T)
    # 方阵是特殊矩阵，A = A.T
    B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
    print("B = B.T:", B == B.T)


def tensor_operation():
    print("张量操作")
    # 张量
    X = torch.arange(24, dtype=torch.int).reshape((2, 3, 4))
    print("张量X:", X)
    # 任何按元素的一元运算都不会改变其操作数的形状
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    # 通过分配新内存，将A的一个副本分配给B
    B = A.clone()
    print("A:", A)
    print("A+B:", A + B)
    # 两个矩阵的按元素乘法称为Hadamard积
    print("Hadamard积:", A * B)


def dimensionality_reduction():
    print("降维")
    x = torch.arange(4, dtype=torch.float32)
    print("x", x, "求和x.sum:", x.sum())
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    # 任意形状张量的元素和
    print("A", A)
    print("shape", A.shape)
    print("sum", A.sum())
    # 指定张量沿哪一个轴来通过求和降低维度 dim=0 =>x轴 1=>y轴
    A_sum_x = A.sum(dim=0)
    print("A_sum_x", A_sum_x)
    print("A_sum_x.shape", A_sum_x.shape)
    A_sum_y = A.sum(dim=1)
    print("A_sum_y", A_sum_y)
    print("A_sum_y.shape", A_sum_y.shape)
    # 所有维度，等价于sum
    A_sum = A.sum(dim=[0, 1])
    print("A_sum", A_sum)
    print("A_sum.shape", A_sum.shape)
    # 平均值（mean或average）
    print("A.mean:", A.mean(), A.sum() / A.numel())
    # 平均值（mean或average）指定维度
    print("A.mean_x:", A.mean(dim=0), A.sum(dim=0) / A.shape[0])
    # 非降维求和,保持轴数不变
    A_sum_y = A.sum(dim=1, keepdim=True)
    print("A_sum_y:", A_sum_y)
    print("A/A_sum_y:", A / A_sum_y)
    # 沿某个轴计算A元素的累积总和
    print("A.cumsum(dim=0):", A.cumsum(dim=0))


def dot_product():
    # 点积 相同位置的按元素乘积的和，可用于加权平均
    x = torch.arange(4, dtype=torch.float32)
    print("x:", x)
    y = torch.ones(4, dtype=torch.float32)
    print("y:", y)
    print("x.dot(y):", torch.dot(x, y))
    # 等价于
    print("x.dot(y):", torch.sum(x * y))


def matrix_vector_product():
    # 矩阵-向量积
    x = torch.arange(4, dtype=torch.float32)
    print("x", x)
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    # 任意形状张量的元素和
    print("A", A)
    # 矩阵-向量积，，A的列维数（沿轴1的长度）必须与x的维数（其长度）相同。
    print("矩阵-向量积", torch.mv(A, x))


def matrix_matrix_multiplication():
    # 矩阵-矩阵乘法
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    # 任意形状张量的元素和
    print("A", A)
    B = torch.ones(4, 3)
    print("B", B)
    torch.mm(A, B)
    # 简单地执行m次矩阵-向量积，并将结果拼接在一起
    print("矩阵-矩阵乘法", torch.mm(A, B))


def norm_operation():
    # 向量范数，相当于距离
    u = torch.tensor([3.0, -4.0])
    print("向量范数, 平方和开根号", torch.norm(u))
    torch.abs(u).sum()
    print("向量范数，绝对值之和", torch.abs(u).sum(), torch.norm(u, p=1))
    print("向量范数，Frobenius范数,平方和的平方根", torch.norm(torch.ones((4, 9))))


if __name__ == '__main__':
    scalar_operation()
    vector_operation()
    matrix_operation()
    tensor_operation()
    dimensionality_reduction()
    dot_product()
    matrix_vector_product()
    matrix_matrix_multiplication()
    norm_operation()
