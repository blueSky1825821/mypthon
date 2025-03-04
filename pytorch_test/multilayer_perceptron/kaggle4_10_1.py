# 如果没有安装pandas，请取消下一行的注释
# !pip install pandas

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

from pytorch_test.multilayer_perceptron.kaggle4_10 import DATA_HUB, DATA_URL, download

"""
1、访问和读取数据集
"""
DATA_HUB['kaggle_house_train'] = (  # @save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  # @save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 训练数据集包括1460个样本，每个样本80个特征和1个标签， 而测试数据集包含1459个样本，每个样本80个特征。
print(train_data.shape)
print(test_data.shape)

# 前四个和最后两个特征，以及相应标签（房价）。
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 在每个样本中，第一个特征是ID， 这有助于模型识别每个训练样本。
# 虽然这很方便，但它不携带任何用于预测的信息。 因此，在将数据提供给模型之前，我们将其从数据集中删除。
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

"""
数据预处理
"""
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
# all_features.dtypes != 'object'：筛选出数据类型不是 object（即非字符串）的列，通常是数值型列。
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    # x.mean()：计算该列的均值。
    # x.std()：计算该列的标准差。
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
# 对分类变量进行独热编码（自动识别非数值列）
# 通过dummy_na=True额外生成列名_nan的布尔列标记缺失值
all_features = pd.get_dummies(all_features, dummy_na=True)
# 独热编码后特征维度扩展（新增列数=原分类特征取值数量+缺失值标记列）
all_features.shape

# 获取训练数据集的行数（样本数量），用于分割特征数据
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 训练
# 损失平方的线性模型
# 均方误差（Mean Squared Error, MSE）损失函数对象。MSE 损失函数用于衡量预测值与真实值之间的差异，常用于回归问题中。
loss = nn.MSELoss()
# 获取该矩阵的第二维度大小，即每个样本的特征数量
in_features = train_features.shape[1]


def get_net():
    """
    函数用于创建并返回一个神经网络模型。
    """
    # nn.Sequential 是一个容器，可以按顺序包含多个层
    # nn.Linear(in_features, 1) 定义了一个线性层（全连接层），输入维度为 in_features，输出维度为 1，即该层将输入特征映射到一个单一的输出值。
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    """
    用于计算预测值和真实值的均方根误差（RMSE），并对预测值进行了对数变换以稳定数值
    """
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    # net(features)：使用神经网络模型 net 对输入特征 features 进行前向传播，得到预测值。
    # torch.clamp(..., 1, float('inf'))：将预测值中小于 1 的部分裁剪为 1，以避免在取对数时出现负无穷或不稳定的数值。float('inf') 表示上界为正无穷。
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    # 对裁剪后的预测值和真实标签分别取自然对数，以稳定数值范围并减少大值的影响。
    # loss(...)：使用之前定义的均方误差损失函数 loss 计算对数变换后的预测值和真实标签之间的 MSE。
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    # 将计算得到的 RMSE 转换为 Python 标量，并返回该标量值。
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """
    训练
    :param net: 神经网络模型。
    :param train_features: 训练集的特征
    :param train_labels:训练集的和标签
    :param test_features: 测试集的特征
    :param test_labels:测试集的标签
    :param num_epochs:  训练的轮数
    :param learning_rate: 学习率
    :param weight_decay: 权重衰减（L2正则化）参数
    :param batch_size: 每个批次的数据量
    :return:
    """
    # train_ls 存储每个epoch后训练集上的 RMSE
    train_ls, test_ls = [], []
    # 创建一个数据迭代器 train_iter，每次迭代返回一个批次的训练数据
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法 对初始学习率不那么敏感,通常收敛更快且更稳定
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()  # 将梯度清零，防止累积。
            l = loss(net(X), y)  # 计算预测值与真实标签之间的损失。
            l.backward()  # 反向传播计算梯度。
            optimizer.step()  # 根据计算的梯度更新模型参数。
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证 有助于模型选择和超参数调整
def get_k_fold_data(k, i, X, y):
    """
    K折交叉过程中返回第i折的数据
    它选择第i个切片作为验证数据，其余部分作为训练数据。 注意，这并不是处理数据的最有效方法，如果我们的数据集大得多，会有其他解决办法。
    :param k: 总共的折数
    :param i: 当前要使用的验证集折的索引
    :param X: 输入特征数据，通常是一个二维张量
    :param y: 标签数据，通常是一个一维张量
    :return: 训练集特征、训练集标签、验证集特征、验证集标签
    """
    assert k > 1
    # X.shape[0]获取输入特征数据的样本数量
    # fold_size是每个折的样本数量，通过总样本数除以 k 得到。
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            # 验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            # 将当前折的数据片段拼接到训练集中
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    """
    在K折交叉验证中训练K次后，返回训练和验证误差的平均值
    :param k: 折数
    :param X_train: 训练集特征数据
    :param y_train: 训练集标签数据
    :param num_epochs: 训练轮数
    :param learning_rate: 学习率
    :param weight_decay: 权重衰减（L2正则化）参数
    :param batch_size: 每个批次的数据量
    :return: 训练和验证误差的平均值
    """
    # 别用于累加所有折的训练和验证误差
    train_l_sum, valid_l_sum = 0, 0
    # K 折循环
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        # 将最后一轮的训练和验证 RMSE 累加到总和中
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            d2l.plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# 模型选择 足够大的数据集和合理设置的超参数，K折交叉验证往往对多次测试具有相当的稳定性。
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
