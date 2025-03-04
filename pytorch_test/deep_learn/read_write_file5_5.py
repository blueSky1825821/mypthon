import torch

from pytorch_test.deep_learn import *

# 保存和读取张量
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print(x2)

# 保存和读取张量列表
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)

# 保存和读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)


# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(self.hidden(X))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# 保存模型参数
torch.save(net.state_dict(), 'mlp.params')
# 加载模型参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
# 设置为评估模式是为了确保模型在推理阶段表现一致，避免训练时的一些随机性影响。
clone.eval()
# 实例具有相同的模型，应该相等
Y_clone = clone(X)
print(Y_clone == Y)
