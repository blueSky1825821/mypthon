"""
填充：在输入图像的边界填充元素（通常填充元素是），避免像素丢失
卷积神经网络中卷积核的高度和宽度通常为奇数，例如1、3、5或7。好处是，
保持空间维度的同时，我们可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列。
"""

from pytorch_test.convolutional_neural_network import *


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    #(1, 1, 8, 8) 为了符合 PyTorch 卷积层的输入格式要求。
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
# 查看初始化后的权重
print("卷积核权重形状:", conv2d.weight.shape)
print("卷积核权重值:\n", conv2d.weight.data)

X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape


"""
步幅（stride）：每次滑动元素的数量
为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素。
"""
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape