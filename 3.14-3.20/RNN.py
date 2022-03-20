# 使用tensflow2
# 1. 读入数据
import pandas as pd
import numpy as np
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train.head())

# 数据有无效的单元
print(np.sum(np.array(train.isnull()==True), axis=0))
print(np.sum(np.array(test.isnull()==True), axis=0))

# fillna 填充处理
train = train.fillna(" ")
test = test.fillna(" ")
print(np.sum(np.array(train.isnull()==True), axis=0))
print(np.sum(np.array(test.isnull()==True), axis=0))

# y标签: 0不是垃圾邮件，1是垃圾邮件
print(train['spam'].unique())  # 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表

# 2. 文本处理
# 邮件内容和主题合并为一个特征
X_train = train['subject'] + ' ' + train['email']
y_train = train['spam']
X_test = test['subject'] + ' ' + test['email']

# 文本转成 tokens ids 序列
from keras.preprocessing.text import Tokenizer
max_words = 300
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
# 只给频率最高的300个词分配 id，其他的忽略
tokenizer.fit_on_texts(list(X_train)+list(X_test)) # tokenizer 训练
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# 样本 tokens 的长度不一样，pad
maxlen = 100
from keras.preprocessing import sequence
X_train_tokens_pad = sequence.pad_sequences(X_train_tokens, maxlen=maxlen,padding='post')
X_test_tokens_pad = sequence.pad_sequences(X_test_tokens, maxlen=maxlen,padding='post')

# 3. 建模
embeddings_dim = 30 # 词嵌入向量维度
from keras.models import Model, Sequential
from keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense
model = Sequential()
model.add(Embedding(input_dim=max_words, # Size of the vocabulary
                    output_dim=embeddings_dim, # 词嵌入的维度
                    input_length=maxlen))
model.add(GRU(units=64)) # 可以改为 SimpleRNN ， LSTM
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

# 4. 训练
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 配置模型
history = model.fit(X_train_tokens_pad, y_train, batch_size=128, epochs=10, validation_split=0.2)
model.save("email_cat_lstm.h5") # 保存训练好的模型

# 绘制曲线
from matplotlib import pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

# 5. 测试
pred_prob = model.predict(X_test_tokens_pad).squeeze()
pred_class = np.asarray(pred_prob > 0.5).astype(np.int32)
id = test['id']
output = pd.DataFrame({'id':id, 'Class': pred_class})
output.to_csv("submission_gru.csv", index=False)


