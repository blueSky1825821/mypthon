from pytorch_test.linear_neural_network import *
from pytorch_test.linear_neural_network import image_classification3_5 as ic


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):  # 即是否为可迭代对象
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):  # 检查 x 是否具有 __len__ 属性。如果不是，则将其复制 n 次，确保 x 和 y 的长度一致
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()  # 清空当前的绘图区域（axes[0]），以便重新绘制新的数据点
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()  # 调用 config_axes 方法配置图表的坐标轴属性，如刻度、标签等。
        display.display(self.fig)  # 显示当前的图表
        display.clear_output(wait=True)  # 清除之前的输出，等待新输出刷新，以实现动态更新效果。


batch_size = 256
# 像素
num_inputs = 28 * 28
# 10个类别
num_outputs = 10
# 生成一个形状为 (784, 10) 的权重矩阵，初始值为均值为 0、标准差为 0.01 的正态分布，并设置为需要计算梯度。
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 生成一个形状为 (10,) 的偏置向量，初始值为全零，并设置为需要计算梯度。
b = torch.zeros(num_outputs, requires_grad=True)

# 只求同一个轴上的元素，即同一列（轴0）或同一行（轴1）
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))


def softmax(X):
    X_exp = torch.exp(X)
    partitions = X_exp.sum(1, keepdim=True)
    return X_exp / partitions  # 这里应用了广播机制


# 依据概率原理，每行总和为1
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))


# reshape函数将每张原始图像展平为向量
def net(X):
    # 其中 -1 表示自动计算行数，W.shape[0] 是权重矩阵 W 的列数
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 这里定义了一个包含两个元素的张量 y，表示两个样本的真实类别索引，分别是 0 和 2
y = torch.tensor([0, 2])
# 这里定义了一个形状为 (2, 3) 的张量 y_hat，表示两个样本在三个类别上的预测概率。每个样本对应一行，每个类别对应一列。
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
"""
这行代码使用高级索引从 y_hat 中提取出每个样本在其真实类别索引处的预测概率。具体来说：
第一个样本（索引 0）的真实类别是 0，因此提取 y_hat[0, 0]，即 0.1。
第二个样本（索引 1）的真实类别是 2，因此提取 y_hat[1, 2]，即 0.5。
最终结果是一个包含两个元素的张量 [0.1, 0.5]，分别表示两个样本在其真实类别上的预测概率。
"""
y_hat[[0, 1], y]


def cross_entropy(y_hat, y):  # 计算交叉熵损失
    return - torch.log(y_hat[range(len(y_hat)), y])


cross_entropy(y_hat, y)


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    """y_hat: 模型的预测结果。
        y: 真实标签。"""
    # 如果 y_hat 是多分类预测（即形状为二维且第二维大于1），则通过 argmax 获取每个样本的最大值索引作为预测类别。
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # 将 y_hat 和 y 进行类型转换并比较，生成布尔数组 cmp。
    cmp = y_hat.type(y.dtype) == y
    # 计算 cmp 中 True 的数量，并将其转换为浮点数返回
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():  # 禁用梯度计算，减少内存消耗并加速计算
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数.如果 updater 是 PyTorch 内置优化器，则调用 zero_grad() ，backward() ，step()
            updater.zero_grad()  # 清空梯度
            l.mean().backward()  # 计算梯度
            updater.step()  # 更新参数
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）
    net：神经网络模型。
    train_iter：训练数据迭代器。
    test_iter：测试数据迭代器。
    loss：损失函数。
    num_epochs：训练的总轮数（epoch）。
    updater：优化器或自定义更新函数。
    """
    # 用于可视化训练过程中的损失和准确率。设置x轴标签为“epoch”
    # x轴范围从1到num_epochs，y轴范围从0.3到0.9，并设置图例为训练损失、训练准确率和测试准确率
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 评估测试集：调用evaluate_accuracy函数评估测试集上的准确率，并将结果存储在test_acc中
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1


def updater(batch_size):
    """
    updater 函数的作用是创建并返回一个用于更新模型参数的小批量随机梯度下降优化器
    :param batch_size: 批量大小。
    :return: 调用 d2l.sgd 函数，传入模型参数 [W, b]、学习率 lr 和批量大小 batch_size，返回一个小批量随机梯度下降（SGD）优化器。
    """
    return d2l.sgd([W, b], lr, batch_size)


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    # 只取一次
    for X, y in test_iter:
        break
    # 将数值标签y转换为文本标签
    trues = d2l.get_fashion_mnist_labels(y)
    # 使用神经网络模型net对输入图像X进行预测，argmax(axis=1)获取每个样本的最大概率对应的类别索引，再将其转换为文本标签。
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    # 将真实标签和预测标签组合成标题，格式为“真实标签\n预测标签”
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    # 展示前n张图像及其对应的标题
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()


def test():
    train_iter, test_iter = ic.load_data_fashion_mnist(batch_size)

    evaluate_accuracy(net, test_iter)
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    predict_ch3(net, test_iter)
