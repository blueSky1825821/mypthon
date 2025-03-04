from pytorch_test.deep_learn import *

# cpu
print(torch.device('cpu'))
# gpu
print(torch.device('cuda'))
# 第几块gpu
print(torch.device('cuda:1'))
# gpu数量
print(torch.cuda.device_count())


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  # @save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


try_gpu(), try_gpu(10), try_all_gpus()

# 张量与GPU，默认情况下，张量在cpu上创建
# 多个项操作时，必须在同一个设备上
x = torch.tensor([1, 2, 3])
print(x.device)

# 存储在GPU
X = torch.ones(2, 3, device=try_gpu())
print(X)
# 第二个GPU上创建一个随机张量
Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)

# 复制数据以在同一设备上执行操作
Z = X.cuda(1)
print(X)
print(Z)
# 数据在同一个GPU上（Z和Y都在）
Y + Z
# 如果已经在一个设备上，不会复制
print(Z.cuda(1) is Z)

# 神经网络与GPU
net = nn.Sequential(nn.Linear(3, 1))
# 模型参数放在GPU
net = net.to(device=try_gpu())
# 模型在同一个GPU上计算结果
net(X)
# 确认模型参数存储在同一个GPU上
print(net[0].weight.data.device)
