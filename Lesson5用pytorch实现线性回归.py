import torch

# prepare dataset
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# design model using class
"""
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called 
"""


class LinearModel(torch.nn.Module): #模型类应该继承自nn.Module，它是所有神经网络的基础类
    def __init__(self):#构造函数，初始化对象时默认调用
        super(LinearModel, self).__init__()#父类的构造函数

        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1) #类torch.nn.Linear后面加括号在构造对象，此类包含两个张量成员：权重和偏置
        #实际上在做运算时，经常把y转置，所以有些公式写成y=wTx+b，都是为了凑维度
    def forward(self, x):#函数的重载：本身模型里有，这里可以说是重写了或者是具体实现了forward
        y_pred = self.linear(x) #linear是一个可调用的对象（由方法__call()__实现，使得对象可以像一个函数一样被调用），实现了wx+b
        return y_pred
    #没有backward，模型自动实现

model = LinearModel()#类的实例化，model是一个可调用的对象

# construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss(reduction='sum') #损失函数：MSE：loss是(y`-y)**2然后求和(可以取平均)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #优化器：model.parameters()自动完成参数的初始化，lr=learning rate

# 训练过程training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)