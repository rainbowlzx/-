import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

#通过框架的内置函数将Fashion-Mnist数据集下载并读取到内存中

trans = transforms.ToTensor() #通过ToTensor将图像数据从PIL数据类型转为32位浮点数格式，并除以255使得所有像素的数值在0与1之间
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
#train=True对应下载训练集，如果是False则对应测试集； transform=trans实现了图片的数据类型转换
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

#len(mnist_train) = 60000, len(mnist_test) = 10000
#mnist_train[0][0].shape = torch.Size([1,28,28])

# 本函数已保存在d2lzh包中方便以后使用
#返回数据集的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

#使用内置的数据迭代器。在每次迭代中，数据加载器每次都会读取一小批量数据，大小为batch_size。我们在训练数据迭代器中还随机打乱了所有样本。
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据。"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()) #num_workers==4

#读取训练数据所需的时间，一般都希望读取时间能比训练时间快一些
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

#整合所有组件。它返回训练集和验证集的数据迭代器。此外，它还接受一个可选参数，用来将图像大小调整为另一种形状。
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

#通过指定resize参数来测试load_data_fashion_mnist函数的图像大小调整功能。
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break