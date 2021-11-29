import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
import torch
from d2l import torch as d2l
import numpy as np

pd.set_option('display.max_columns', 100) #最多显示100列数据

'''读取数据'''
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# print(train_data.head(5))

'''数据预处理'''
# 删除样本id和state，并将训练集及测试集拼在一起处理（此处的训练集比测试集多了一列Sold Price，拼接后测试集这列会留空）
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
'''
# 我自己进行的筛选，虽然训练集上表现很好，但预测结果分数很低，大量有用的特征没有使用。泛化能力差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # 获取数值特征的索引
# print(numeric_features)
# iloc的参数是行号或者列号，loc的参数是行标签或者列标签
all_features = all_features.loc[:,['Sold Price', 'Year built', 'Lot', 'Bathrooms', 'Full bathrooms',
                                   'Total interior livable area', 'Total spaces', 'Garage spaces',
                                   'Elementary School Score', 'Elementary School Distance',
                                   'Middle School Score', 'Middle School Distance', 'High School Score',
                                   'High School Distance', 'Tax assessed value', 'Annual tax amount',
                                   'Listed Price', 'Last Sold Price', 'Zip']] # 保留筛选出来的所有数值特征
# print(all_features.head(5))

# 计算相关性，通过热力图展示
cor = all_features.corr()
sns.heatmap(cor, annot = True, cmap=plt.cm.Reds)
#plt.show()

# 筛选出于自变量与因变量之间的相关性
cor_target = abs(cor["Sold Price"])
# 挑选于大于0.05的相关性系数，最终选择了12个特征
relevant_features = cor_target[cor_target>0.05] # 类型是Series，类似元组（标签，相关系数）
# print(relevant_features)
all_features = all_features.loc[:,['Sold Price', 'Bathrooms', 'Full bathrooms', 'Elementary School Score',
                                   'Elementary School Distance', 'Middle School Score', 'Middle School Distance',
                                   'High School Score', 'High School Distance', 'Tax assessed value',
                                   'Annual tax amount', 'Listed Price', 'Last Sold Price']]

# 可以考虑要不要再看自变量之间的相关性进行筛选

# 标准化数据，对所有列进行，所以不用再限制列
all_features = all_features.apply(lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有数据均值消失，因此我们可以将缺失值设置为0
all_features = all_features.fillna(0)
# print(all_features.head(5))
'''

all_features = all_features.loc[:, ["Year built", "Lot", "Bathrooms", "Full bathrooms", "Total spaces",
                                    'Elementary School Score', 'Middle School Score',"High School Score",
                                    "Tax assessed value", "Annual tax amount", "Listed Price",
                                    "Last Sold Price", "Zip"]] # 在网络版本的基础上添加了几个我筛选出的特征，结果略有提高
# 处理数值特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 字符串特征，则用one-hot编码替换
all_features = pd.get_dummies(all_features, dummy_na=True) # pd.get_dummies是：利用pandas实现one hot encode的方式

# 切分回训练集和测试集，从pandas格式中提取NumPy格式，并将其转换为张量表示
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data["Sold Price"].values.reshape(-1, 1), dtype=torch.float32)
print(train_features.shape)
print(train_labels.shape)


'''训练'''
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 32),
                        nn.ReLU(),
                        nn.Linear(32,16),
                        nn.ReLU(),
                        nn.Linear(16,1))
    return net

# 用价格预测的对数来衡量差异
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1。torch.clamp(input, min, max, out=None)
    # 将输入input张量每个元素的夹紧到区间[min,max]，并返回结果到一个新张量。
    clipped_preds = torch.clamp(net(features), 1, float('inf')) #?
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
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

#K折交叉验证。有助于模型选择和超参数调整。需要一个函数，在K折交叉验证过程中返回第i折的数据。它选择第i个切片作为验证数据，其余部分作为训练数据。
#每一次切分数据为测试集和验证集
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


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.01, 0.1, 64
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
    plt.show()
    print(f'train log rmse {float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)

#一种良好的完整性检查是查看测试集上的预测是否与 K 倍交叉验证过程中的预测相似。如果是，那就是时候把它们上传到Kaggle了。
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
