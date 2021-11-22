import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# 定义函数将类别标签转为id表示，方便后面计算交叉熵
def lables2id(lables):
    target_id = []
    target_lables = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    for lable in lables:
        target_id.append(target_lables.index(lable))
    return target_id


# 定义数据集类
class ProductDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        lables = data['target']
        self.len = data.shape[0]  # shape(多少行，多少列)

        self.x_data = torch.tensor(np.array(data)[:, 1:-1].astype(float))
        self.y_data = lables2id(lables)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = ProductDataset('train.csv')

# 建立数据集加载器
train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(93, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.linear4 = torch.nn.Linear(16, 9)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            x = self.relu(self.linear3(x))
            x = self.relu(self.linear4(x))
            # 这里先取出最大概率的索引，即是所预测的类别。
            _, predicted = torch.max(x, dim=1)
            # 将预测的类别转为one-hot表示，方便保存为预测文件。
            y = pd.get_dummies(predicted)
            return y


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

loss_list=[]

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


# 开始训练
if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)

plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 定义预测保存函数，用于保存预测结果。
def predict_save():
    test_data = pd.read_csv('test.csv')
    test_inputs = torch.tensor(np.array(test_data)[:, 1:].astype(float))
    out = model.predict(test_inputs.float())

    lables = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    # 添加列标签
    out.columns = lables
    # 插入id行
    out.insert(0, 'id', test_data['id'])
    output = pd.DataFrame(out)
    output.to_csv('my_predict.csv', index=False)


predict_save()
