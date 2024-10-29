import torch
from pytorch_test import *

"""
微积分
"""


def calc_f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(func, x, h):
    # 导数函数
    return (func(x + h) - func(x)) / h


def derivative():
    # 导数计算
    h = 0.1
    for i in range(5):
        print(f'h={h:.5f}, numerical limit={numerical_lim(calc_f, 1, h):.5f}')
        h *= 0.1


def use_svg_display():  # @save
    """使用svg格式在Jupyter中显示绘图"""
    # @save是一个特殊的标记，会将对应的函数、类或语句保存在d2l包中。
    # matplotlib软件包输出svg图表以获得更清晰的图像。
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


# @save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    d2l.plt.show()

def draw():
    x = np.arange(0, 3, 0.1)
    plot(x, [calc_f(x), 2 * x - 3], 'x', 'calc_f(x)', legend=['calc_f(x)', 'Tangent line (x=1)'])


if __name__ == '__main__':
    derivative()
    draw()
