import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 穷举法
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1): #生成列表：strat,stop,step
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data): #zip：可有多个参数。返回一个以元组为元素的列表，其中第i个元组包含每个参数序列的第i个元素。
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)
#可视化
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
