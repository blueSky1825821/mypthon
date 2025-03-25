"""
批量规范化：持续加深深层网络的收敛速度
"""
import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    批量规范化
    :param X: 输入张量，形状为 [batch_size, num_features] 或 [batch_size, channels, height, width]。
    :param gamma:  缩放参数，用于调整标准化后的数据范围。
    :param beta: 偏移参数，用于调整标准化后的数据偏移。
    :param moving_mean: 移动平均均值，用于预测模式下的标准化。
    :param moving_var: 移动平均方差，用于预测模式下的标准化。
    :param eps: 平滑项，防止除以零。eps 是一个小值（如 (1e-5)），用于避免分母为零。
    :param momentum: 动量参数，用于更新移动平均均值和方差.gamma 和 beta 是可学习的参数，用于恢复数据的表达能力。
    :return:
    """
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # X 是一个张量，通常表示一批输入数据（例如形状为 [batch_size, num_features]）。
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            # dim=0 表示对每个特征（列）计算均值，忽略批次维度（行）。结果 mean 是一个形状为 [num_features] 的一维张量，表示每列特征的均值。
            mean = X.mean(dim=0)
            # 一个形状为 [num_features] 的一维张量，表示每列特征的方差。
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 功能: 计算张量 X 在指定维度 (0, 2, 3) 上的均值，并保持维度形状不变。
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            # dim=(0, 2, 3): 表示对批次维度（dim=0）、高度维度（dim=2）和宽度维度（dim=3）进行均值计算。
            # 换句话说，对于每个通道（channels），计算该通道所有像素的均值。
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data
