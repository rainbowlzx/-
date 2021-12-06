import random
import torch
from d2l import torch as d2l


tokens = d2l.tokenize(d2l.read_time_machine()) # 把这本书按行拆成单词，tokens=[[第一行第一词,第一行第二词...],[第二行第一词...]...]
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line] # corpus=[第一行第一词,第一行第二词...第二行第一词...]
vocab = d2l.Vocab(corpus) # 构建词汇表。用来将字符串类型的词元映射到从0开始的数字索引中。
print(vocab.token_freqs[:10]) # 打印出现次数最多的10个词

# 事实上最流行的词看起来很无聊。通常被称为停用词（stop words），因此可以被过滤掉。但它们本身仍然是有意义的，仍然会在模型中使用它们。
# 此外，还有个明显的问题是词频衰减的速度相当地快。
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
# d2l.plt.show()

# 二元语法
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])] #zip实现了二元语法的打包，序列标号为(1,2),(2,3)...(n-1,n)
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

# 三元语法
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

# 直观对比三种模型中的词元频率
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
# d2l.plt.show()


# 随机采样
def seq_data_iter_random(corpus, batch_size, num_steps):  # num_steps 是每个子序列中预定义的时间步数tau
    """使用随机抽样生成一个小批量子序列。"""
    # 从随机偏移量开始对序列进行分区，随机范围包括`num_steps - 1`，前面会丢掉一点数据。这样使得每次切的位置都不同，能得到不同的序列
    # 好处是遍历一次，每个数字只用一次，而不是之前讲的用tau次
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签。看有多少段子序列。
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为`num_steps`的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从`pos`位置开始的长度为`num_steps`的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，`initial_indices`包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch] # y是对应的下一个x，即希望预测的下一位
        yield torch.tensor(X), torch.tensor(Y)

#生成一个从0到34的序列。批量大小为2，时间步数为5，这意味着可以生成⌊(35−1)/5⌋=6个“特征－标签”子序列对。能得到3个小批量。
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

# 顺序分区。除了对原始序列可以随机抽样外，我们还可以保证两个相邻的小批量中的子序列在原始序列上也是相邻的。
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列。"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

#将上面的两个采样函数包装到一个类中，以便稍后可以将其用作数据迭代器。
class SeqDataLoader:  #@save
    """加载序列数据的迭代器。"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

# 定义了一个函数，它同时返回数据迭代器和词汇表
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词汇表。"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab