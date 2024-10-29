from pytorch_test import *

"""
抽样sampling：从概率分布中抽取样本的过程
分布：笼统的可以把分布看成对事件的概率分配
多项分布multinomial distribution：概率分配给一些离散选择的分布
"""

fair_probs = torch.ones([6]) / 6
print(fair_probs)
# 抽样
sample = multinomial.Multinomial(1, fair_probs).sample()
print(sample)
sample = multinomial.Multinomial(10, fair_probs).sample()
print(sample)
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts)
# 相对频率作为估计值
print(counts / 1000)

# 500次实验
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# x累加
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()
