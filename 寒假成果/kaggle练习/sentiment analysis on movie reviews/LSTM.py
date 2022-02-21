# 因为没有安装tensorflow而报错
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

# 处理数据
# 由于是文本数据，并且已经分词完毕，所以就先对文本进行去除空格，标点符号和全部变成小写。
def dataCleaning(totalText):
    total = []
    for i in totalText:
        temp = i.lower()
        temp = re.sub('[^a-zA-Z]', ' ', temp)
        tempArr = [j for j in temp.strip().split('\t') if isinstance(j, str)]
        tstr = ' '.join(tempArr)
        total.append(tstr)
    return total


def loadData(name):
    data = pd.read_csv(name, delimiter='\t')
    totalText = data['Phrase']
    totalText = dataCleaning(totalText)
    totalLabel = data['Sentiment']
    return totalText, totalLabel


def getTest(name):
    data = pd.read_csv(name, delimiter='\t')
    totalText = data['Phrase']
    totalText = dataCleaning(totalText)
    return totalText


# 加载数据集
totalText, totalLabel = loadData('../train.tsv.zip')
testText = getTest('../test.tsv.zip')  # 路径写你自己的路径

# 查看数据
print(len(totalText))
print(len(totalLabel))
print(len(testText))

# 查看评分的类别分布
uniqueLabel = set(totalLabel)
x = []
y = []
for i in uniqueLabel:
    x.append(i)
    y.append(totalLabel[totalLabel == i].size)
plt.figure(111)
plt.bar(x, y)
plt.xlabel('type of review  ')
plt.ylabel('count')
plt.title('Movie Review')
plt.show()

# tokenizer
train_tokenizer = Tokenizer()
train_tokenizer.fit_on_texts(totalText)
train_sequences = train_tokenizer.texts_to_sequences(totalText)
test_sequences = train_tokenizer.texts_to_sequences(testText)

# 获得所有tokens的长度
num_tokens = [len(tokens) for tokens in train_sequences]
num_tokens = np.array(num_tokens)
print(len(num_tokens))
# 输出  156060

# 平均tokens的长度
print('mean', np.mean(num_tokens))
# 最长的评价tokens的长度
print('max', np.max(num_tokens))
# 最长的评价tokens的长度
print('min', np.min(num_tokens))

# 查看训练数据的长度
plt.hist((num_tokens), bins=50)
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)
# 取tokens的长度为19时，大约93%的样本被涵盖

np.sum(num_tokens < max_tokens) / len(num_tokens)

# padding
train_Data = sequence.pad_sequences(train_sequences, maxlen=48)
test_Data = sequence.pad_sequences(test_sequences, maxlen=48)
# 这里的长度可以是前面取到的max_tokens，也可以是他的最大长度  前者可以节约计算时间

print(train_Data.shape)
print(test_Data.shape)
print(train_Data)

train_label = to_categorical(totalLabel, 5)
print(train_label.shape)
print(train_label)

# 转换标签.多类分类问题与二类分类问题类似，需要将类别变量（categorical function）的输出标签转化为数值变量。用one hot encoding方法
train_label = to_categorical(totalLabel, 5)
print(train_label.shape)
print(train_label)

# 在测试样本上切分数据集和测试集
X_train, X_test, y_train, y_test = train_test_split(train_Data, train_label, test_size=0.25, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 使用Keras来搭建LSTM。max_features为最多单词数
max_features = len(train_tokenizer.index_word)
max_len = 48  # 这个是要和前面padding时的长度一致
epochs = 5  # 训练次数
emb_dim = 128  # 128代表embedding层的向量维度
batch_size = 80  # 这是指定批量的大小

# LSTM搭建
model = Sequential()

model.add(Embedding(max_features, emb_dim, mask_zero=True))
model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# 验证一下
model.evaluate(X_test, y_test)

# 预测测试集
predict = model.predict_classes(test_Data)
sub = pd.read_csv('../input/sampleSubmission.csv')
sub['Sentiment'] = predict
sub.to_csv('sub_lstm.csv', index=False)
