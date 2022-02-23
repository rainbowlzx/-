# 导入数据集
# pandas是用来导入、整理、清洗表格数据的专用工具，类似excel，但功能更加强大，导入的时候给pandas起个小名叫pd
import pandas as pd
# 用pandas的read_csv函数读取训练数据及测试数据，数据文件是.tsv格式的，也就是说数据用制表符\t分隔，类似于.csv文件的数据用逗号分隔
data_train = pd.read_csv('./train.tsv.zip',sep='\t')
data_test = pd.read_csv('./test.tsv.zip',sep='\t')
# 看训练集数据前5行，Phrase列为电影评论文本，Sentiment为情感标签
print(data_train.head())
# 共有156060行训练数据，每行数据都有短语ID、句子ID、文本内容、情感标签四列
print(data_train.shape)

# 查看测试集数据前5行，Phrase列就是需要我们自己构建模型预测情感标签的文本
print(data_test.head())
# 共有66292行测试集数据，每个数据都有短语ID、句子ID、文本内容三列
print(data_test.shape)

# 构建语料库
# 我们需要对文本进行一些处理，将原始文本中的每一个词变成计算机看得懂的向量，这一过程叫做文本的特征工程，非常重要。
# 有很多将词变成向量的方法，比如下面将要介绍的词袋模型、TF-IDF模型，以及视频中介绍的word2vec模型。
# 不管采用什么模型，我们都需要先把训练集和测试集中所有文本内容组合在一起，构建一个语料库。
# 提取训练集中的文本内容
train_sentences = data_train['Phrase']

# 提取测试集中的文本内容
test_sentences = data_test['Phrase']

# 通过pandas的concat函数将训练集和测试集的文本内容合并到一起
sentences = pd.concat([train_sentences,test_sentences])
# 合并到一起的语料库共有222352行数据
print(sentences.shape)
# 提取训练集中的情感标签，一共是156060个标签
label = data_train['Sentiment']
print(label.shape)

# 导入停词库，停词库中的词是一些废话单词和语气词，对情感分析没什么帮助
stop_words = open('D:/pycharm/Project/stop words.txt',encoding='utf-8').read().splitlines()
# stop_words是一个列表，列表中每一个元素都是一个停用词

# 用sklearn库中的CountVectorizer构建词袋模型
# analyzer='word'指的是以词为单位进行分析，对于拉丁语系语言，有时需要以字母'character'为单位进行分析
# ngram指分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
# max_features指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词

from sklearn.feature_extraction.text import CountVectorizer
co = CountVectorizer(analyzer='word',ngram_range=(1,4),stop_words=stop_words,max_features=150000)

# 使用语料库，构建词袋模型
co.fit(sentences)

# 将训练集随机拆分为新的训练集和验证集，默认3:1,然后进行词频统计
# 在机器学习中，训练集相当于课后习题，用于平时学习知识。验证集相当于模拟考试，用于检验学习成果。测试集相当于高考，用于最终Kaggle竞赛打分。
# 新的训练集和验证集都来自于最初的训练集，都是有标签的
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_sentences,label,random_state=1234)

# 用上面构建的词袋模型，把训练集和验证集中的每一个词都进行特征工程，变成向量
x_train = co.transform(x_train)
x_test = co.transform(x_test)
# 随便看训练集中的一个数据，它是150000列的稀疏矩阵

# 构建分类器算法，对词袋模型处理后的文本进行机器学习和数据挖掘

# 直接运行本单元格即可，本单元格代码的作用是：忽略下面代码执行过程中的版本警告等无用提示
import warnings
warnings.filterwarnings('ignore')

# 使用逻辑回归分类器
from sklearn.linear_model import LogisticRegression
lg1 = LogisticRegression()
lg1.fit(x_train,y_train)
print('词袋方法进行文本特征工程，使用sklearn默认的逻辑回归分类器，验证集上的预测准确率:',lg1.score(x_test,y_test))

#引用朴素贝叶斯进行分类训练和预测
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print('词袋方法进行文本特征工程，使用sklearn默认的多项式朴素贝叶斯分类器，验证集上的预测准确率:',classifier.score(x_test,y_test))


