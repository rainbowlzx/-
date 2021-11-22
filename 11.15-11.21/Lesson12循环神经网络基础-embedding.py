import torch

num_class = 4           #类别数量
input_size = 4          #输入维度
hidden_size = 8         #隐藏层维度
embedding_size = 10     #嵌入到10维空间
num_layers = 2          #RNN层数
batch_size = 1
seq_len = 5             #数据量
idx2char = ['e', 'h', 'l', 'o']  # 方便最后输出结果
x_data = [[1, 0, 2, 2, 3]]  # (batch, seq_len)
y_data = [3, 1, 2, 3, 2]  # (batch * seq_len)

inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x) #input (batch,seqLen)  output (batch, seqLen, embeddingSize)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class) #修改维度好用交叉熵计算损失

net = Model()
#---计算损失和更新
criterion = torch.nn.CrossEntropyLoss()#交叉熵
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
#---计算损失和更新

for epoch in range(15):#训练15次
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)##从第一个维度上取出预测概率最大的值和该值所在序号
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
