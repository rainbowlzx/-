#问题在于文字特征的处理，内存跑不动了
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

#用的不是加州数据，因为object里字符太多，one-hot编码处理那里会有问题
train_data = pd.read_csv('D:/pycharm/Project/Pytorch深度学习实践/homework'
                         '/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('D:/pycharm/Project/Pytorch深度学习实践/homework'
                        '/house-prices-advanced-regression-techniques/test.csv')

#删除每个样本的id，并用concat连结训练集和测试集数据，首尾相连，即竖着拼接各表
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

#数据预处理
#将所有缺失的值替换为相应特征的平均值。为了将所有特征放在一个共同的尺度上，通过将特征重新缩放到零均值和单位方差来标准化数据。
#获取数据类型不为object的特征的索引，即获取所有数值的索引；DataFrame中字符串的存储类型是object，不是str
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有数据均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#处理离散值。用一次独热编码替换。例如，“MSZoning”包含值“RL”和“Rm”。将创建两个新的指示器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。
# `Dummy_na=True` 将“na”（缺失值）视为有效的特征值，并为其创建指示符特征。
all_features = pd.get_dummies(all_features, dummy_na=True)
#all_features.shape (2919, 331)


#此转换会将特征的数量从79个增加到331个。最后，通过values属性，我们可以从pandas格式中提取NumPy格式，并将其转换为张量表示用于训练。
n_train = train_data.shape[0] #看训练数据表有多少行，即有多少个训练样本
train_features = torch.tensor(all_features[:n_train].values, dtype=d2l.float32) #切分回去训练集和测试集，转换为张量
test_features = torch.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)

#训练
#线性模型不会在竞赛中获胜，但提供了一种健全性检查，以查看数据中是否存在有意义的信息。如果不能做得比随机猜测更好，那么很可能存在数据处理错误。
loss = nn.MSELoss() #均方损失
in_features = train_features.shape[1] #看训练数据表有多少列，即有多少输入特征

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

#对于房价，就像股票价格一样，关心的是相对数量，而不是绝对数量。因此，更关心相对误差而不是绝对误差。
#解决这个问题的一种方法是用价格预测的对数来衡量差异。事实上，这也是比赛中官方用来评价提交质量的误差指标。公式参见4.10.2
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1。torch.clamp(input, min, max, out=None)
    # 将输入input张量每个元素的夹紧到区间[min,max]，并返回结果到一个新张量。
    clipped_preds = torch.clamp(net(features), 1, float('inf')) # inf：正无穷，float()变成浮点数
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

#K折交叉验证。一般用于模型调优，找到使得模型泛化性能最优的超参值。，找到后，在全部训练集上重新训练模型，并使用独立测试集对模型性能做出最终评价。
#需要一个函数，在K折交叉验证过程中返回第i折的数据。它选择第i个切片作为验证数据，其余作为训练数据。每一次切分数据为测试集和验证集
def get_k_fold_data(k, i, X, y):
    assert k > 1 #assert断言：如果不成立程序在此会直接报错。方便查找程序的问题。
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # slice(start,stop,step)函数实现切片对象，主要用在切片操作函数里的参数传递。start处的值被切掉了，如果是0则不切
        idx = slice(j * fold_size, (j + 1) * fold_size) #第j次切块，切出了第j块数据集
        X_part, y_part = X[idx, :], y[idx] #按索引切出验证集的数据
        if j == i: #i是整体操作进行到第几次了；第i次时，取第i块的那些数据作为验证集，其他块的数据作为训练集
            X_valid, y_valid = X_part, y_part
        elif X_train is None: #训练集空时直接赋值，不为空时则将切分出来的数据与它拼起来
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0) #cat:按维数0把X_train, X_part拼起来（竖着拼）
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

#在K折交叉验证中训练K次后，返回训练和验证误差的平均值。
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):#一共验证k次：划分数据，训练，验证
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

#原始超参数为5, 100, 5, 0, 64，结果0.16
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 3, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}')

#训练所有数据，提交预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

#一种良好的完整性检查是查看测试集上的预测是否与 K 倍交叉验证过程中的预测相似。如果是，那就是时候把它们上传到Kaggle了。
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)