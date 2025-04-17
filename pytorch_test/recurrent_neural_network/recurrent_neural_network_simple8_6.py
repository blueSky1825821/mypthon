import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

#定义模型
#RNN 模型中隐藏层的单元数（hidden units）。隐藏单元的数量决定了模型的表达能力：更大的隐藏单元数量可以捕获更复杂的模式，但也会增加计算复杂度和内存消耗。
num_hiddens = 256
#表示输入词汇表的大小。每个时间步的输入是一个 one-hot 向量，其维度等于词汇表的大小。
rnn_layer = nn.RNN(len(vocab), num_hiddens)

#张量初始化隐状态 （隐藏层数，批量大小，隐藏单元数）
state = torch.zeros((1, batch_size, num_hiddens))
state.shape

#生成一个指定形状的张量，其元素是从 [0, 1) 范围内的均匀分布中随机采样的。
#num_steps：表示时间步的数量（即序列长度）
#len(vocab)：表示词汇表的大小。每个时间步的输入是一个长度为 len(vocab) 的向量（通常是 one-hot 编码或嵌入向量）。
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
#rnn_layer的“输出”（Y）不涉及输出层的计算： 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入。
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape

#@save
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

#训练与预测
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)

num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)