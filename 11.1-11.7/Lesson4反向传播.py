import torch
#tensor是张量，可以存储标量、矢量、3维甚至更高维度的数据，包含数据本身+梯度，例如tersor w包括w的值和loss对w的梯度
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # w的初值为1.0
w.requires_grad = True  # 需要计算梯度，要设置，写上这一句即可


def forward(x):
    return x * w  # w是一个Tensor，x也会重载为tensor，结果是一个tensor


def loss(x, y): #在构建计算图，每次调用都会产生一个计算图
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()  # backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w.grad.item()) #item也是防止产生计算图，拿出标量
        w.data = w.data - 0.01 * w.grad.data  # 权重更新时，需要用到标量（grad也是一个tensor）。不用建立计算图来求整个过程，所以只用标量
        #item只能对标量操作，data返回的其实也是一个tensor，只不过不计算梯度
        w.grad.data.zero_()  # after update, remember set the grad to zero
        #不清0梯度就会一直求和下去
    print('progress:', epoch, l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("predict (after training)", 4, forward(4).item())
