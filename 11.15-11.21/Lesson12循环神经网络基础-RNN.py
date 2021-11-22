import torch

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [ [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1] ]
x_one_hot = [one_hot_lookup[x] for x in x_data]

input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)

class RNNModel(torch.nn.Module):
    def __init__(self,input_size, hidden_size, batch_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)  # 提供初始化隐藏层（h0）
        out, _ = self.rnn(input, hidden)#out=[ h0, h1, h2, h3, h4]  _ = [[[h4]]]
        return out.view(-1, self.hidden_size) #Reshape out to(seqLen*batchSize,hiddenSize)

net = RNNModel(input_size, hidden_size,batch_size, num_layers)
#---计算损失和更新
criterion = torch.nn.CrossEntropyLoss()#交叉熵
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
#---计算损失和更新

for epoch in range(100):#训练100次
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)##从第一个维度上取出预测概率最大的值和该值所在序号
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/100] loss = %.3f' % (epoch + 1, loss.item()))
