import collections
import re
from d2l import torch as d2l

# 读取数据集
# 从 H.G.Well 的《时光机器》这本书中中加载文本
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f: # r表示读文件，with会在之后自动调用close，代码更简洁
        lines = f.readlines() #按行读取
    # 把非字母的字符都变成空格，比如标点符号；strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列;lower把大写变成小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])

# 词元化。
# 列表中的每个元素是一个文本序列。每个文本序列又被拆分成一个词元列表，词元（token）是文本的基本单位。
# 返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）。
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元。"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])

# 词汇表
# 词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。让我们构建一个字典，通常也叫做词汇表（vocabulary），
# 用来将字符串类型的词元映射到从0开始的数字索引中。 我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计，
# 得到的统计结果称之为语料（corpus）。然后根据每个唯一词元的出现频率，为其分配一个数字索引。很少出现的词元(min_freq)通常被移除，可以降低复杂性。
# 语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元 “<unk>” 。 我们可以选择增加一个列表(reserved_tokens)，
# 用于保存那些被保留的词元，例如：填充词元（“<pad>”）；序列开始词元（“<bos>”）；序列结束词元（“<eos>”）。
class Vocab:  #@save
    """文本词汇表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # 按出现频率从大到小排列
        # 未知词元的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    # 给token，可以返回它的下标index
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    # 给下标，可以返回token（这两个函数实现的功能正好相反）
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):  #@save
    """统计词元的频率。"""
    # 这里的 `tokens` 是 1D 列表或 2D 列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成使用词元填充的一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens) # 计数器

#首先使用时光机器数据集作为语料库来构建词汇表，然后打印前几个高频词元及其索引。
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

# 可以将每一条文本行转换成一个数字索引列表。
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])

# 整合所有功能
# 函数返回 corpus（词元索引列表）和 vocab（时光机器语料库的词汇表）。
# 改变： 1. 为了简化后面章节中的训练，使用字符（而不是单词）实现文本词元化；
# 2. 数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的corpus仅处理为单个列表，而不是使用多个词元列表构成的一个列表。
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词汇表。"""
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
print(len(corpus)) #是一个列表，存储的是这本书按字母转换成的一列索引数字
print(len(vocab)) # 28 = 26个字母 + <unk> + 空格