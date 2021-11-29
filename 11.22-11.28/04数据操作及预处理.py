import torch

x=torch.arange(12)
print(x)

print(x.shape) #访问张量的形状
print(x.numel()) #访问张量中元素的总数

x=x.reshape(3,4) #只改变张量的形状，不改变元素数量和元素值
print(x)

print(torch.zeros(2,3,4)) #使用全0
print(torch.ones(2,3,4)) #使用全1

print(torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]]).shape) #使用列表来为张量中的每一个元素赋予确定值
print(torch.tensor([[[2,1,4,3],[1,2,3,4],[4,3,2,1]]]).shape) #多了一个中括号，注意维度区别

#常见的标准运算符都可以升级为按元素运算
a=torch.tensor([1,2,3,4])
b=torch.tensor([2,2,2,2])
print(a+b)
print(a*b)

#可以把多个张量连结在一起
x=torch.arange(12,dtype=torch.float32).reshape((3,4))
y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((x,y),dim=0)) #横向连结
print(torch.cat((x,y),dim=1)) #纵向连结

#通过逻辑运算符构建二元张量
print(x==y)

print(x.sum())

#广播机制
c=torch.arange(3).reshape((3,1))
d=torch.arange(2).reshape((1,2))
print(c+d)

#转换为NumPy张量
A=x.numpy()
B=torch.tensor(A)
print(type(A))
print(type(B))

#将大小为1的张量转换为Python标量
z=torch.tensor([3.5])
print(z)
print(z.item())
print(float(z))
print(int(z))

#tensor默认float64，但是深度学习中计算较慢，一般用float32