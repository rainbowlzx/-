import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# prepare dataset


class DiabetesDataset(Dataset):#继承自Dataset类
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):#支持通过下标索引拿出data；在数据集很大时，保证高效使用内存，只在需要时根据下标读取对应数据，不用一次全读
        return self.x_data[index], self.y_data[index]

    def __len__(self): #返回数据集的长度
        return self.len


dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # num_workers 多线程
#batch_size：一次前向反向中训练样本的个数；shuffle是打乱顺序；num_workers到底用几个并行化程序读取数据

# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
if __name__ == '__main__':#是由于windows系统在使用多进程的时候用spawn，会出现问题，所以只能将这些封装起来（可封装为函数）
    epoch_list = []
    loss_list = []
    for epoch in range(1000): #epoch：所有的训练样本经过一次前向和反向传播为一次epoch
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch，0是指从0开始枚举
            #enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据（写在后）和数据下标（写在前）：（索引，数据）
            #enumerate(sequence, [start=0])
            #1.prepare data
            inputs, labels = data #inputs和labels都是张量，自动转换
            #2.forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            epoch_list.append(epoch)
            loss_list.append(loss.item())
            #3.backward
            optimizer.zero_grad()
            loss.backward()
            #4.update
            optimizer.step()
    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
