import collections
import re
from d2l import torch as d2l

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中
    with 语句确保在代码块执行完毕后自动关闭文件。
    """
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
        #使用正则表达式 [^A-Za-z]+ 匹配非英文字母的字符（如标点符号、数字等），并将它们替换为空格。
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        #line.split() 的作用是按照空格将字符串拆分为单词列表。
        return [line.split() for line in lines]
    elif token == 'char':
        #list(line) 的作用是将字符串拆分为单个字符的列表。
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


class Vocab:  #@save
    """文本词表
    类用于构建文本数据的词表（vocabulary）。它将文本中的词元映射为唯一的索引值，并支持以下功能：
    按词元频率排序。
    支持保留特定的词元（如 <pad>、<bos> 等）。
    忽略低频词元。
    提供词元到索引和索引到词元的双向映射。
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        :param tokens: 词元列表，可以是一维或二维列表。默认为空列表。
        :param min_freq: 最低词频阈值，低于该值的词元会被忽略。默认为 0。
        :param reserved_tokens: 需要保留的特殊词元（如 <pad>、<bos> 等）。默认为空列表。
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        #并按频率从高到低排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0 存储词元的顺序列表，索引为词元的唯一标识。
        self.idx_to_token = ['<unk>'] + reserved_tokens
        #存储词元到索引的映射字典。
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率
    可以是一个一维列表（如 ['word1', 'word2']），也可以是一个二维列表（如 [ ['word1', 'word2'], ['word3'] ]）
    """
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])


for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

def load_corpus_time_machine(max_tokens=-1):
    """
    :param max_tokens: 限制返回的词元数量。默认值为 -1，表示不限制词元数量
    """
    #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)