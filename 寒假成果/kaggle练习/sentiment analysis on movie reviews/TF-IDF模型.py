# dual那里一直报错

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

import warnings
warnings.filterwarnings('ignore')

# 用sklearn库中的TfidfVectorizer构建TF-IDF模型
# analyzer='word'指的是以词为单位进行分析，对于拉丁语系语言，有时需要以字母'character'为单位进行分析
# ngram指分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
# max_features指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词

# TF-IDF模型是专门用来过滤掉烂大街的词的，所以不需要引入停用词stop_words

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_features=150000)
print(tf.fit(sentences))

# 类似上面的操作，拆分原始训练集为训练集和验证集，用TF-IDF模型对每一个词都进行特征工程，变成向量
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_sentences,label,random_state=1234)
x_train = tf.transform(x_train)
x_test = tf.transform(x_test)

# 构建分类器算法，对TF-IDF模型处理后的文本进行机器学习和数据挖掘
# 朴素贝叶斯分类器
# 引用朴素贝叶斯进行分类训练和预测
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print('TF-IDF方法进行文本特征工程，使用sklearn默认的多项式朴素贝叶斯分类器，验证集上的预测准确率:',classifier.score(x_test,y_test))
# TF-IDF方法进行文本特征工程，使用sklearn默认的多项式朴素贝叶斯分类器，验证集上的预测准确率: 0.6045367166474432

# 逻辑回归分类器
# sklearn默认的逻辑回归模型
from sklearn.linear_model import LogisticRegression
lg1 = LogisticRegression()
lg1.fit(x_train,y_train)
print('TF-IDF方法进行文本特征工程，使用sklearn默认的逻辑回归模型，验证集上的预测准确率:',lg1.score(x_test,y_test))
# TF-IDF方法进行文本特征工程，使用sklearn默认的逻辑回归模型，验证集上的预测准确率: 0.6326541073945918
# C：正则化系数，C越小，正则化效果越强
# dual：求解原问题的对偶问题
lg2 = LogisticRegression(C=3, dual=True)
lg2.fit(x_train,y_train)
print('TF-IDF方法进行文本特征工程，使用增加了两个参数的逻辑回归模型，验证集上的预测准确率:',lg2.score(x_test,y_test))
# TF-IDF方法进行文本特征工程，使用增加了两个参数的逻辑回归模型，验证集上的预测准确率: 0.6533384595668332
# 对比两个预测准确率可以看出，在逻辑回归中增加C和dual这两个参数可以提高验证集上的预测准确率，但如果每次都手动修改就太麻烦了。
# 我们可以用sklearn提供的强大的网格搜索功能进行超参数的批量试验。
# 搜索空间：C从1到9。对每一个C，都分别尝试dual为True和False的两种参数。最后从所有参数中挑出能够使模型在验证集上预测准确率最高的。

from sklearn.model_selection import GridSearchCV
param_grid = {'C':range(1,10),'dual':[True,False]}
lgGS = LogisticRegression()
grid = GridSearchCV(lgGS, param_grid=param_grid,cv=3,n_jobs=-1)
grid.fit(x_train,y_train)

print(grid.best_params_)

lg_final = grid.best_estimator_
print('经过网格搜索，找到最优超参数组合对应的逻辑回归模型，在验证集上的预测准确率:',lg_final.score(x_test,y_test))

# 对测试集的数据进行预测，提交Kaggle竞赛最终结果
# 查看测试集数据前5行，Phrase列就是需要我们自己构建模型预测情感标签的文本
print(data_test.head())
# 使用TF-IDF对测试集中的文本进行特征工程
test_X = tf.transform(data_test['Phrase'])
# 对测试集中的文本，使用lg_final逻辑回归分类器进行预测
predictions = lg_final.predict(test_X)

# 将预测结果加在测试集中
data_test.loc[:,'Sentiment'] = predictions

# 按Kaggle比赛官网上的要求整理成这样的格式
final_data = data_test.loc[:,['PhraseId','Sentiment']]

# 保存为.csv文件，即为最终结果
final_data.to_csv('final_data.csv',index=None)