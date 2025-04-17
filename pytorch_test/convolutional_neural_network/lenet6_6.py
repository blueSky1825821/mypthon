from pytorch_test.convolutional_neural_network import *

"""
卷积神经网络
"""

"""
LeNet
卷积编码器：由两个卷积层组成;
全连接层密集块：由三个全连接层组成。
"""
net = nn.Sequential(
    # 添加一个二维卷积层。输入通道数为1（如灰度图像），输出通道数为6，使用5x5的卷积核，并在输入的每一边填充2个像素。这有助于保持输入和输出的空间尺寸一致。
    # 添加一个Sigmoid激活函数层。它将上一层的输出值映射到0到1之间，增加模型的非线性能力。
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # 添加一个平均池化层。使用2x2的窗口大小和步长为2进行下采样，减少特征图的尺寸，同时保留重要信息。
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 再次添加一个二维卷积层。这次输入通道数为6（上一层的输出通道数），输出通道数为16，使用5x5的卷积核。这里没有指定padding，默认为0。
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 将多维张量展平成一维向量，以便连接到全连接层。
    # 卷积层的输出通常是四维张量（batch_size, channels, height, width），而全连接层需要二维张量（batch_size, features）。
    nn.Flatten(),
    # 全连接层，输入维度为16 * 5 * 5（来自卷积层的展平输出），输出维度为120。
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

"""
模型训练
"""
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        # 将模型设置为评估模式。这会关闭一些只在训练时启用的操作（如Dropout和BatchNorm），以确保评估时的行为一致。
        # Dropout 行为：在评估模式下，Dropout层不会丢弃任何神经元，所有神经元都参与前向传播，且输出不被缩放。
        # 作用：稳定输出：确保模型在评估或推理时的输出是确定性的，避免因随机性导致的结果波动。
        # BatchNorm 行为：在评估模式下，BatchNorm层不再使用当前小批量的数据来计算均值和方差，而是使用训练过程中累积的运行平均值和运行方差来进行归一化。
        # 作用：一致性和稳定性：确保在评估或推理时，模型的输出是基于整个训练集的统计信息，而不是单个小批量数据，从而使结果更加稳定和可靠。
        net.eval()  # 设置为评估模式
        # 如果device参数未指定（即为None），则自动获取模型中第一个参数所在的设备（通常是GPU或CPU）。
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    # 在此上下文中禁用梯度计算。因为评估时不涉及反向传播和参数更新，禁用梯度可以节省内存并加速计算。
    with torch.no_grad():
        for X, y in data_iter:
            # 检查输入数据X是否为列表。如果X是列表（例如BERT微调任务中可能有多个输入张量），则将每个元素移动到指定设备上。
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # 获取当前批次中样本的数量。
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# @save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    """
    net: 神经网络模型。
    train_iter: 训练数据迭代器。
    test_iter: 测试数据迭代器。
    num_epochs: 训练的总轮数（epoch）。
    lr: 学习率。
    device: 指定使用的设备（如GPU或CPU）。
    """

    def init_weights(m):
        """
        定义一个内部函数，用于初始化模型中的线性层和卷积层的权重。
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # 使用Xavier均匀分布初始化该层的权重，以确保权重的初始值合理，有助于加速训练过程。
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    # 将模型net移动到指定的设备上（如GPU），以便利用其计算能力加速训练。
    net.to(device)
    # 创建一个随机梯度下降（SGD）优化器，用于更新模型参数。lr是学习率。
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 创建交叉熵损失函数，用于计算预测值与真实标签之间的误差。
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    # num_batches 获取训练数据迭代器中批次的数量。
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        # 将模型设置为训练模式，启用一些仅在训练时使用的操作（如Dropout和BatchNorm）。
        # Dropout 行为：在训练模式下，Dropout层会随机丢弃一部分神经元（即将其输出设置为0），并将其余神经元的输出按比例放大以保持期望值不变。
        # 作用:1)防止过拟合：通过随机丢弃部分神经元，迫使网络学习更鲁棒的特征表示，减少对特定神经元的依赖。2)正则化：相当于对模型进行多次不同的“稀疏”训练，从而提高模型的泛化能力。
        # BatchNorm:行为：在训练模式下，BatchNorm层使用当前小批量（mini-batch）的数据来计算均值和方差，并对其进行归一化处理。同时，它还会维护一个运行平均值（running mean）和运行方差（running variance），用于后续的评估阶段。
        # 作用：1)加速收敛：通过归一化每层输入，使各层的输入分布更加稳定，从而加快训练速度2)缓解梯度消失/爆炸问题：归一化操作有助于保持梯度在合理范围内，避免梯度过大或过小。3)增强模型稳定性：使模型对初始权重和超参数的选择更加鲁棒。
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()  # 启动计时器，记录训练时间。
            optimizer.zero_grad()  # 清空模型参数的梯度，避免梯度累积。
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()  # 对损失值l调用backward()方法，计算梯度。
            optimizer.step()  # 根据计算出的梯度更新模型参数。
            with torch.no_grad():  # 在此上下文中禁用梯度计算，因为评估时不涉及反向传播和参数更新，禁用梯度可以节省内存并加速计算。
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 不更新梯度
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
