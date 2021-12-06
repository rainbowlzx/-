# 从零开始实现
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35 # num_steps一条样本有多长，T
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# one-hot编码
# print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

# 小批量数据形状是：批量大小*时间步长=32*35
X = torch.arange(10).reshape((2, 5))
# print(F.one_hot(X.T, 28).shape) # X.T转置，维度是：时间、批量、每个样本的特征长度（时间步数, 批量大小, 词汇表大小）
                                # 能够更方便地通过最外层的维度，一步一步地更新小批量数据的隐藏状态。

# 初始化参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens)) # 这是RNN与之前讨论的网络最主要的区别
    b_h = torch.zeros(num_hiddens, device=device) # 偏置初始化0
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 在初始化时返回隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), ) # 做成了tuple，主要是为了和之后的LSTM等统一规范

# 如何在一个时间步内计算隐藏状态和输出
def rnn(inputs, state, params):
    # `inputs`的形状：(`时间步数量`，`批量大小`，`词表大小`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state # state是一个元组，只不过这里只有第一个维度有值，后面是空的。见上一个函数
    outputs = []
    # `X`的形状：(`批量大小`，`词表大小`)
    for X in inputs: # input:(时间、批量、词表)，会按照第一个维度去遍历，每一步算一个特定的时间步，拿出时刻0的（批量、词表）、时刻1的...
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h) # 输入层到隐藏层
        Y = torch.mm(H, W_hq) + b_q # 隐藏层到输出层
        outputs.append(Y) # 把所有时刻的结果记在一个表里
    return torch.cat(outputs, dim=0), (H,) # dim=0按行连接，即竖着拼。结果是列数不变，行数=时间步数*批量大小。返回输出和隐藏状态

# 创建一个类来包装这些函数，并存储从零开始实现的循环神经网络模型的参数。
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 检查输出是否具有正确的形状
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
# print(Y.shape, len(new_state), new_state[0].shape) #共需要预测10=2*5（批量大小*时间维度）个值，每个值是28类中的一种
# 输出形状是（时间步数 × 批量大小，词汇表大小），而隐藏状态形状保持不变，即（批量大小, 隐藏单元数）

# 定义预测函数来生成prefix之后的新字符，其中的prefix是一个用户提供的包含多个字符的字符串。在循环遍历prefix中的开始字符时，
# 我们不断地将隐藏状态传递到下一个时间步，但是不生成任何输出。这被称为“预热”（warm-up）期，因为在此期间模型会自我更新（例如，更新隐藏状态），
# 但不会进行预测。预热期结束后，隐藏状态的值通常比刚开始的初始值更适合预测，从而预测字符并输出它们。
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在`prefix`后面生成新字符。"""
    state = net.begin_state(batch_size=1, device=device) # 初始化最开始的状态
    outputs = [vocab[prefix[0]]] # 把第一个词从vocab中拿到对应的下标，放到output
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1)) # 最后预测的那个词作为下一时刻的输入
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state) # 不在乎输出，只是为了初始化状态
        outputs.append(vocab[y]) # 存的是真实值，而不是预测值，避免累计误差
    for _ in range(num_preds):  # 预测`num_preds`步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1))) # 拿出概率最大的那个，转成标量，保存
    return ''.join([vocab.idx_to_token[i] for i in outputs]) # 把下标转成字符

# 测试predict_ch8函数。我们将前缀指定为time traveller ，并基于这个前缀生成10个后续字符。鉴于我们还没有训练网络，它会生成荒谬的预测结果。
# print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

def grad_clipping(net, theta):  #@save
    """裁剪梯度。"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params)) # 所有的一起操作
    if norm > theta:  # 如果大了，就裁剪（映射），梯度范数永远不会超过θ
        for param in params:
            param.grad[:] *= theta / norm


#训练
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期（定义见第8章）。"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和, 词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter: # 如果是随机迭代，每个时间输入的序列之间没有时序关系，不能共用状态，所以要初始化0
            # 在第一次迭代或使用随机抽样时初始化`state`
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else: # 按顺序迭代的话，不用每次都初始化；backward只在一次iteration里面，做梯度时不关心之前的计算图，就删掉不要
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state`对于`nn.GRU`是个张量
                state.detach_()
            else:
                # `state`对于`nn.LSTM`或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1) # 先转置，再拉成一个向量
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state) # forward
        l = loss(y_hat, y.long()).mean() # y.long() tensor数据类型转换，浮点型转成长整型。这也解释了之前的cat(dim=0)
                                         # 对于loss而言就是一个多分类问题，只不过实际的批量大小变成了设置的批量大小*时间步数
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1) # 梯度剪裁
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了`mean`函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop() # 困惑度：正常的loss做了一个指数

#也可以使用高级API实现
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）。"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

#训练循环神经网络模型
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

# 检查一下随机抽样的结果
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)