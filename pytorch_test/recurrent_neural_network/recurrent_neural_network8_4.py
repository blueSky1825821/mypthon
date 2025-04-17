import torch
import math

from d2l import torch as d2l

from torch import nn
from torch.nn import functional as F

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)

## 上面等价于下面
#沿列（轴1）拼接矩阵X和H， 沿行（轴0）拼接矩阵W_xh和W_hh。 这两个拼接分别产生(3,5)形状和(5,4)形状的矩阵。
# 再将这两个拼接的矩阵相乘， 我们得到与上面相同形状(3,4)的输出矩阵。

torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))


batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

F.one_hot(torch.tensor([0, 2]), len(vocab))

X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape


def get_params(vocab_size, num_hiddens, device):
    """
    初始化循环神经网络（RNN）的参数。
    :param vocab_size: 输入和输出的词表大小，表示输入数据的维度和输出预测的类别数。
    :param num_hiddens: 隐藏层的单元数量，决定了隐藏状态的维度。
    :param device: 指定参数存储的设备（如 CPU 或 GPU）
    :return:
    """
    #将输入维度 num_inputs 和输出维度 num_outputs 都设置为 vocab_size。
    #因为在语言模型中，输入和输出通常都是一组独热编码向量，其维度等于词表大小。
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    """

    :param inputs: 形状为 (时间步数量, 批量大小, 词表大小) 的张量。
                    每个时间步的输入是一个形状为 (批量大小, 词表大小) 的矩阵，表示当前时间步的输入数据。
    :param state: 初始隐藏状态，形状为 (批量大小, 隐藏单元数)。
                    在 RNN 中，隐藏状态用于存储从之前时间步传递下来的信息。
    :param params: W_xh: 输入到隐藏层的权重矩阵，形状为 (词表大小, 隐藏单元数)。
                    W_hh: 隐藏层到隐藏层的权重矩阵，形状为 (隐藏单元数, 隐藏单元数)。
                    b_h: 隐藏层的偏置项，形状为 (隐藏单元数,)
                    W_hq: 隐藏层到输出层的权重矩阵，形状为 (隐藏单元数, 输出单元数)。
                    b_q: 输出层的偏置项，形状为 (输出单元数,)。
    :return:
    """
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X: 当前时间步的输入，形状为 (批量大小, 词表大小)。
    for X in inputs:
        #torch.mm(X, W_xh)：输入与输入到隐藏层的权重相乘。
        #torch.mm(H, W_hh)：上一时间步的隐藏状态与隐藏层到隐藏层的权重相乘。
        #b_h：隐藏层的偏置项
        #使用 torch.tanh 激活函数对结果进行非线性变换。
        #隐藏层激活：在循环神经网络（RNN）中，torch.tanh 常用于隐藏层的激活函数。它能够将加权输入和偏置的线性组合映射到非线性空间，从而使模型能够学习更复杂的模式。
        #平滑梯度：相比于其他激活函数（如 ReLU），tanh 的输出范围是连续的，有助于在网络训练过程中平滑梯度，避免梯度消失或梯度爆炸问题。
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        #torch.mm(H, W_hq)：隐藏状态与隐藏层到输出层的权重相乘。
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
        #使用 torch.cat(outputs, dim=0) 将所有时间步的输出按维度 0（时间步维度）拼接成一个整体输出张量。
        #最终输出的形状为 (时间步数量 * 批量大小, 输出单元数)。
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape

def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """
    预测：该函数适用于基于 RNN 的文本生成任务。通过提供一个前缀字符串，模型可以自动生成后续的字符序列，从而实现文本续写或生成的功能。
    :param prefix: 一个字符串，表示用户提供的初始文本片段。模型会根据这个前缀生成后续的字符
    :param num_preds: 需要生成的新字符数量。
    :param net:
    :param vocab: 词表对象，包含字符到索引和索引到字符的映射。
    :param device:
    :return:
    """
    """在prefix后面生成新字符"""
    #调用 net.begin_state 方法初始化隐藏状态。
    #设置批量大小为 1，因为预测时只处理单个样本。
    state = net.begin_state(batch_size=1, device=device)
    #将前缀的第一个字符转换为对应的索引值，并存储在 outputs 列表中。
    outputs = [vocab[prefix[0]]]
    #定义一个匿名函数 get_input，用于将 outputs 列表中的最后一个字符索引转换为形状为 (1, 1) 的张量。
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期：处理前缀字符串,使其隐藏状态能够捕捉到前缀字符串的上下文信息。
        #调用 net 模型，传入当前字符的索引张量和隐藏状态，更新隐藏状态。
        _, state = net(get_input(), state)
        #将当前字符的索引值添加到 outputs 列表中。
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步,循环 num_preds 次，每次生成一个新字符
        #调用 net 模型，传入当前字符的索引张量和隐藏状态，得到输出张量 y 和更新后的隐藏状态。
        y, state = net(get_input(), state)
        #使用 y.argmax(dim=1) 找到输出张量中概率最大的索引值，表示预测的下一个字符。因为使用的one_hot
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())

def grad_clipping(net, theta):  #@save
    """
    是一个用于梯度裁剪（Gradient Clipping）的函数。在训练循环神经网络（RNN）时，由于序列长度较长或模型复杂度较高，可能会出现梯度爆炸（Gradient Explosion）的问题。
    梯度爆炸会导致模型参数更新不稳定，从而影响训练效果。梯度裁剪是一种常用的解决方案，通过限制梯度的大小来避免这一问题。
    :param net:
    :param theta:
    :return:
    """
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    #遍历所有参数 params，计算每个参数梯度的平方和。
    #使用 torch.sqrt 对平方和开平方，得到梯度的 L2 范数（即梯度的欧几里得范数）。
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """
    是一个用于训练循环神经网络（RNN）模型一个迭代周期的函数。它通过遍历训练数据集，计算损失并更新模型参数，同时记录训练过程中的指标（如困惑度和速度）
    :param net: RNN 模型对象
    :param train_iter: 训练数据迭代器，提供批量训练数据 (X, Y)，其中 X 是输入序列，Y 是目标序列。
    :param loss: 失函数，用于衡量模型预测值与真实值之间的差异。
    :param updater: 参数更新方法，可以是 PyTorch 的优化器（如 torch.optim.SGD），也可以是从零开始实现的自定义更新函数。
    :param device: 指定计算设备（如 CPU 或 GPU）
    :param use_random_iter: 表示是否使用随机抽样生成训练数据。如果为 True，则在每次迭代时重新初始化隐藏状态。
    :return:
    """
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    #获取每个批次的输入序列 X 和目标序列 Y。
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                #将隐藏状态中的梯度分离（detach_()），以避免梯度回传到之前的隐藏状态，从而减少内存消耗。
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        #将目标序列 Y 转置并展平为一维张量 y，形状为 (-1,)。
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            #清空梯度。
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        #如果 updater 是自定义更新函数：
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        #累加损失值 l * y.numel() 和词元数量 y.numel() 到累积器 metric 中。
        metric.add(l * y.numel(), y.numel())
    #计算困惑度：math.exp(metric[0] / metric[1])，表示平均损失的指数。
    #计算训练速度：metric[1] / timer.stop()，表示每秒处理的词元数量。
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """
    使用高级API实现
    """
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

#使用随机抽样方法
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)