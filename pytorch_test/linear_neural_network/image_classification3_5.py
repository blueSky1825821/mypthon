from pytorch_test.linear_neural_network import *

d2l.use_svg_display()

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../../data", train=False, transform=trans, download=True)
print(len(mnist_train), len(mnist_test))

# 每个输入图像的高度和宽度均为28像素，数据集由灰度图像组成，通道数为1
print(mnist_train[0][0].shape)


def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """
    :param imgs: 图像列表
    :param num_rows: 行数
    :param num_cols: 列数
    :param titles: 每个图像的标题列表（可选）
    :param scale: 图像缩放比例，默认为1.5
    :return:
    """
    """绘制图像列表"""
    """计算图像的总尺寸"""
    figsize = (num_cols * scale, num_rows * scale)
    """创建子图网格 axes轴对象"""
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    """展平轴对象，将多维的axes展平为一维数组"""
    axes = axes.flatten()
    """"遍历图像列表：使用 enumerate 遍历 axes 和 imgs，同时获取索引 i、轴对象 ax 和图像 img"""
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        """判断图像是否为张量（torch.is_tensor(img)），如果是张量则转换为 numpy 数组并显示，否则直接显示 PIL 图像"""
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        """隐藏坐标轴：隐藏每个子图的 x 轴和 y 轴。"""
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        """设置标题：如果提供了标题列表 titles，则设置当前图像的标题"""
        if titles:
            ax.set_title(titles[i])
    """返回轴对象：返回所有子图的轴对象 axes"""
    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.show()

"""读取小批量"""
batch_size = 256


def get_dataloader_workers():  # @save
    """使用4个进程来读取数据"""
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

"""读取训练数据需要的时间"""
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """
    用于获取和读取Fashion-MNIST数据集
    :param batch_size: 读取批量
    :param resize: 用来将图像大小调整为另一种形状
    :return: 训练集和验证集的数据迭代器。
    """
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    """将输入的 PIL 图像或 numpy 数组转换为 PyTorch 的 Tensor，并且将像素值从 [0, 255] 范围归一化到 [0, 1] 范围"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    """将所有变换组合成一个单一的变换对象，以便后续应用于数据集中的每个图像"""
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
"""X 数据 y标签"""
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
