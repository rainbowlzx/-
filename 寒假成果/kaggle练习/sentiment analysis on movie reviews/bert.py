# 报错原因是内存不够，电脑跑不动
from datetime import timedelta
import torch.nn as nn
import pandas as pd
import numpy as np
import torch

# 读取数据集
train_csv = pd.read_csv('./train.tsv.zip',sep='\t')
test_csv = pd.read_csv('./test.tsv.zip',sep='\t')
# print(train_csv.head(5)) 可以查看前5行
# print(test_csv.head(5))

# 划分出验证集
np.random.shuffle([train_csv])
rows,_ = train_csv.shape
valid_data = train_csv.iloc[int(rows*0.7):,:]
train_data = train_csv.iloc[:int(rows*0.7),:]
# print(valid_csv.head(5))

X_train = train_data['Phrase']
y_train = train_data['Sentiment']
X_valid = valid_data['Phrase']
y_valid = valid_data['Sentiment']
X_test = test_csv['Phrase']
y_test = [0] * len(X_test)  # 测试集没有标签，这么处理方便代码处理
y_test = torch.LongTensor(y_test)  # 转成tensor

# 设置超参数
PAD, CLS = '[PAD]', '[CLS]'
max_seq_len = 64
bert_hidden = 768
num_classes = 5
learning_rate = 1e-5
decay = 0.01
num_epochs = 5
early_stop_time = 2000  # 最早停止时间：当准确度无法再提高且达到此时间时，结束训练过程；在收敛后至少再训练2000个batch才结束当前epoch
batch_size = 32
save_path = "./best_model.ckpt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 导入bert预训练模型
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")  # 使用预训练好的模型实现词元化

pretrain_model = BertForSequenceClassification.from_pretrained("./bert-base-uncased", num_labels=num_classes)

def load_dataset(texts, labels):  # 按bert模型要求加载数据
    contents = []
    for t, label in zip(texts, labels):
        token = tokenizer.tokenize(t)
        token = [CLS] + token
        # ['[CLS]', 'subject', ':', 'cell', 'phones', 'coming', 'soon', '<', 'html', '>', ...]
        seq_len = len(token)
        mask = []
        token_ids = tokenizer.convert_tokens_to_ids(token)
        # [101, 3395, 1024, 3526, 11640, 2746, 2574, 1026, 16129, 。。。]
        if len(token) < max_seq_len:
            mask = [1] * len(token) + [0] * (max_seq_len - len(token)) # mask用1来标识哪些位置是原有词汇，0来哪些位置是补的0
            token_ids = token_ids + [0] * (max_seq_len - len(token)) # 长度不够时，尾端补0
        else:
            mask = [1] * max_seq_len
            token_ids = token_ids[:max_seq_len] # 长度超长时，裁去尾部
            seq_len = max_seq_len
        y = [0] * num_classes
        y[label] = 1
        contents.append((token_ids, y, seq_len, mask))
    return contents


load_dataset(X_train, y_train)


def build_iter(datasets, batch_size, device):  # 整合数据，建立迭代器
    iter = datasetIter(datasets, batch_size, device)
    return iter

# 数据集迭代器类
class datasetIter():
    def __init__(self, datasets, batch_size, device):
        self.datasets = datasets
        self.idx = 0  # 表示将要处理第几组
        self.device = device
        self.batch_size = batch_size
        self.batches = len(datasets) // batch_size  # batches记录一共有多少组
        self.residues = False  # 用residues记录是否全部按整组遍历完，False表示没有剩余
        if len(datasets) % batch_size != 0:
            self.residues = True

    def __next__(self):
        if self.residues and self.idx == self.batches:  # 最后一组不满 且 将要处理最后一组
            batch_data = self.datasets[self.idx * self.batch_size: len(self.datasets)] # 将当前batch的数据切出来
            self.idx += 1
            batch_data = self._to_tensor(batch_data)
            return batch_data
        elif self.idx > self.batches:  # 全部组都处理完了
            self.idx = 0
            raise StopIteration
            # 当调用iter函数的时候，生成了一个迭代对象，要求__iter__必须返回一个实现了__next__的对象，
            # 就可以通过next函数访问这个对象的下一个元素了，并且在不想继续有迭代的情况下抛出一个StopIteration的异常
        else: # 处理的是除最后一组外的其他组（数量都是满的）
            batch_data = self.datasets[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
            self.idx += 1
            batch_data = self._to_tensor(batch_data)
            return batch_data

    def _to_tensor(self, datasets):  # 用device将数据转换成tensor
        x = torch.LongTensor([item[0] for item in datasets]).to(self.device)
        y = torch.FloatTensor([item[1] for item in datasets]).to(self.device)
        seq_len = torch.LongTensor([item[2] for item in datasets]).to(self.device)
        mask = torch.LongTensor([item[3] for item in datasets]).to(self.device)
        return (x, seq_len, mask), y

    def __iter__(self):
        return self
    # 用__iter__和__next__这两种函数的好处：每次产生的数据，是产生一个用一个，什么意思呢，比如我要遍历[0, 1, 2, 3.....]，
    # 如果使用列表的方式，那么是会全部载入内存的。但是如果使用迭代器，可以看见，当用到了(也就是在调用了next)才会产生对应的数字，节约内存

    def __len__(self):
        if self.residues:
            return self.batches + 1
        else:
            return self.batches


class myModel(nn.Module):  # 模型类
    def __init__(self):
        super(myModel, self).__init__()
        self.pretrain_model = pretrain_model
        for param in self.pretrain_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        context = x[0]
        mask = x[2]
        out = self.pretrain_model(context, attention_mask=mask)  # 先套用预训练模型
        out = torch.softmax(out.logits, 1)  # 最后使用一层softmax；out.logits手动使用sigmoid/softmax将之前网络的输入映射到[0,1]
        return out


# %%

import time
import torch.nn.functional as F

from sklearn import metrics
from transformers.optimization import AdamW


def get_time_dif(starttime):
    # calculate used time
    endtime = time.time()
    return timedelta(seconds=int(round(endtime - starttime)))


def evaluate(model, dev_iter):  # 评估函数，计算准确度和损失
    model.eval()
    loss_total = 0
    pred_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for X, y in dev_iter:
            outputs = model(X)
            # y = y.unsqueeze(1)
            loss = F.binary_cross_entropy(outputs, y)
            loss_total += loss
            truelabels = torch.max(y.data, 1)[1].cpu()
            pred = torch.max(outputs, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, truelabels)
            pred_all = np.append(pred_all, pred)
    acc = metrics.accuracy_score(labels_all, pred_all)
    return acc, loss_total / len(dev_iter)


def test(model, test_iter):
    model.load_state_dict(torch.load(save_path))
    model.eval()  # 训练集在train下运行，验证集和测试集都在eval下运行（解释详见train()）
    pred_all = np.array([], dtype=int)
    with torch.no_grad():
        for X, y in test_iter:
            outputs = model(X)
            pred = torch.max(outputs, 1)[1].cpu().numpy()
            pred_all = np.append(pred_all, pred)
    id = test_csv['PhraseId']
    output = pd.DataFrame({'PhraseId': id, 'Sentiment': pred_all})
    output.to_csv("submission_bert.csv", index=False)


def train(model, train_iter, dev_iter, test_iter): # 汇总整个过程
    starttime = time.time()
    model.train()  # 如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。
    # model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    total_batch = 0  # 记录已经训练了多少epoch
    dev_best_loss = float("inf")
    last_improve = 0
    no_improve_flag = False
    model.train()
    for epoch in range(num_epochs):  # 一个epoch是把整个训练集数据跑一遍
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        for i, (X, y) in enumerate(train_iter):
            outputs = model(X)  # batch_size * num_classes # 前向传播
            model.zero_grad()  # 梯度清零
            loss = F.binary_cross_entropy(outputs, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            if total_batch % 100 == 0:  # 每100batch计算一次训练集和测试集的准确度
                truelabels = torch.max(y.data, 1)[1].cpu()  # 实际值。方括号[1]表示返回最大值的索引
                pred = torch.max(outputs, 1)[1].cpu()  # 预测值
                train_acc = metrics.accuracy_score(truelabels, pred)
                dev_acc, dev_loss = evaluate(model, dev_iter)  # 同样地计算验证集的准确度和损失，但不更新梯度，所以单用一个函数写
                if dev_loss < dev_best_loss:  # 是否更新最小损失
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ' '
                time_dif = get_time_dif(starttime)
                # 打印训练信息，id : >右对齐，n 宽度，.3 小数位数
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, ' \
                      'val Acc :{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > early_stop_time:  # 收敛后又训练了够长的时间，结束训练（没收敛或者时间不够，都继续）
                print("no improve after {} times, stop!".format(early_stop_time))
                no_improve_flag = True
                break
        if no_improve_flag:
            break
    test(model, test_iter)  # 进行测试


# 开始训练
np.random.seed(520)
torch.manual_seed(520)
torch.cuda.manual_seed_all(520)
torch.backends.cudnn.deterministic = True

train_data = load_dataset(X_train, y_train)
valid_data = load_dataset(X_valid, y_valid)
test_data = load_dataset(X_test, y_test)
train_iter = build_iter(train_data, batch_size, device)
valid_iter = build_iter(valid_data, batch_size, device)
test_iter = build_iter(test_data, batch_size, device)

model = myModel().to(device) # 模型实例化
train(model, train_iter, valid_iter, test_iter)
