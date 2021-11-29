from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns  #统计数据可视化库
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, f_regression,chi2
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest

'''处理连续型的特征'''
x = load_boston() #此函数可以直接读数据
df = pd.DataFrame(x.data, columns = x.feature_names)
df["MEDV"] = x.target
X = df.drop("MEDV",1)   #将模型当中要用到的特征变量保留下来
y = df["MEDV"]          #最后要预测的对象
#df.head() 根据位置返回对象的前n行。此处没写n，返回所有行

#计算一下自变量和因变量之间的相关性，通过seaborn模块当中的热力图来展示
plt.figure(figsize=(10,8)) #设置图片大小
cor = df.corr() #计算相关系数。相关系数接近于0意味着变量之间的相关性并不强，接近于-1意味着负相关，接近于1意味着正相关
#annot默认为false，true会在图上显示数字；cmapmatplotlib：colormap name or object, or list of colors, optional
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# 筛选出于自变量与因变量之间的相关性
cor_target = abs(cor["MEDV"])
# 挑选于大于0.5的相关性系数
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

#看一下自变量之间的相关性如何，要是自变量之间的相关性非常强的话，我们也只需要保留其中的一个就行，
print(df[["LSTAT","PTRATIO"]].corr())
print("=" * 50)
print(df[["RM","LSTAT"]].corr())
print("=" * 50)
print(df[["PTRATIO","RM"]].corr())
#RM变量和LSTAT这个变量是相关性是比较高的，只需要保留其中一个就可以了。选择保留LSTAT这个变量，因为它与因变量之间的相关性更加高一些

#递归消除法：选择一个基准模型，起初将所有的特征变量传进去，在确认模型性能的同时通过对特征变量的重要性进行排序，去掉不重要的特征变量
#然后不断地重复上面的过程直到达到所需数量的要选择的特征变量。
LR= LinearRegression()
# 挑选出7个相关的变量
rfe_model = RFE(LR, n_features_to_select=7) #此处由于新版本的库的函数参数格式改了，所以与公众号上略有不同
# 交给模型去进行拟合
X_rfe = rfe_model.fit_transform(X,y)
LR.fit(X_rfe,y)
# 输出各个变量是否是相关的，并且对其进行排序
print(rfe_model.support_)
print(rfe_model.ranking_)
#第一行的输出包含True和False，其中True代表的是相关的变量对应下一行的输出中的1，而False包含的是不相关的变量

#将13个特征变量都依次遍历一遍
feature_num_list=np.arange(1,13)
# 定义一个准确率
high_score=0
# 最优需要多少个特征变量
num_of_features=0
score_list =[]
for n in range(len(feature_num_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe_model = RFE(model,n_features_to_select=feature_num_list[n])
    X_train_rfe_model = rfe_model.fit_transform(X_train,y_train)
    X_test_rfe_model = rfe_model.transform(X_test)
    model.fit(X_train_rfe_model,y_train)
    score = model.score(X_test_rfe_model,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        num_of_features = feature_num_list[n]
print("最优的变量是: %d个" %num_of_features)
print("%d个变量的准确率为: %f" % (num_of_features, high_score))

#看一下到底是哪些特征变量
cols = list(X.columns)
model = LinearRegression()
# 初始化RFE模型，筛选出10个变量
rfe_model = RFE(model, n_features_to_select=10)
X_rfe = rfe_model.fit_transform(X,y)
# 拟合训练模型
model.fit(X_rfe,y)
df = pd.Series(rfe_model.support_,index = cols)
selected_features = df[df==True].index
print(selected_features)

#正则化
#对于Lasso的正则化而言，对于不相关的特征而言，该算法会让其相关系数变为0，因此不相关的特征变量很快就会被排除掉了，只剩下相关的特征变量
lasso = LassoCV()
lasso.fit(X, y)
#Series 是带有标签的一维数组，可以保存任何数据类型（整数，字符串，浮点数，Python对象等）,轴标签统称为索引
coef = pd.Series(lasso.coef_, index = X.columns)

#然后我们看一下哪些变量的相关系数是0
print("Lasso算法挑选了 " + str(sum(coef != 0)) + " 个变量，然后去除掉了" +  str(sum(coef == 0)) + "个变量")

#对计算出来的相关性系数排个序并且做一个可视化
imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (8, 6)
imp_coef.plot(kind = "barh")
plt.title("Lasso Model Feature Importance")
plt.show()


'''处理离散型的特征'''
'''此部分找不到原始数据!!!
#根据缺失值来判断：如果一个特征绝大部分的值都是缺失的，就没有必要考虑了
#首先导入所需要用到的数据集
train = pd.read_csv("credit_example.csv")
train_labels = train['TARGET']
train = train.drop(columns = ['TARGET'])

#先计算一下数据集当中每个特征变量缺失值的比重
missing_series = train.isnull().sum() / train.shape[0]
df = pd.DataFrame(missing_series).rename(columns = {'index': '特征变量', 0: '缺失值比重'})
df.sort_values("缺失值比重", ascending = False).head()

#output
                           缺失值比重
COMMONAREA_AVG            0.6953
COMMONAREA_MODE           0.6953
COMMONAREA_MEDI           0.6953
NONLIVINGAPARTMENTS_AVG   0.6945
NONLIVINGAPARTMENTS_MODE  0.6945

#我们可以看到缺失值最高的比重将近有70%，我们也可以用可视化的根据来绘制一下缺失值比重的分布图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.figure(figsize = (7, 5))
plt.hist(df['缺失值比重'], bins = np.linspace(0, 1, 11), edgecolor = 'k', color = 'blue', linewidth = 2)
plt.xticks(np.linspace(0, 1, 11))
plt.xlabel('缺失值的比重', size = 14)
plt.ylabel('特征变量的数量', size = 14)
plt.title("缺失值分布图", size = 14)
#我们可以看到有一部分特征变量，它们缺失值的比重在50%以上，有一些还在60%以上，我们可以去除掉当中的部分特征变量


#计算特征的重要性
#在基于树的众多模型当中，会去计算每个特征变量的重要性，也就是feature_importances_属性，得出各个特征变量的重要性程度之后再进行特征的筛选
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
# 模型拟合数据
clf.fit(X,Y)
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
# 筛选出特征的重要性程度最大的10个特征
feat_importances.nlargest(10)

#我们同时也可以对特征的重要性程度进行可视化，
feat_importances.nlargest(10).plot(kind='barh', figsize = (8, 6))
#除了随机森林之外，基于树的算法模型还有很多，如LightGBM、XGBoost等等，大家也都可以通过对特征重要性的计算来进行特征的筛选
'''

#Select_K_Best算法
#在Sklearn模块当中还提供了SelectKBest的API，针对回归问题或者是分类问题，我们挑选合适的模型评估指标，
#然后设定K值也就是既定的特征变量的数量，进行特征的筛选。
#假定我们要处理的是分类问题的特征筛选，我们用到的是iris数据集

iris_data = load_iris()
x = iris_data.data
y = iris_data.target

print("数据集的行与列的数量: ", x.shape)

#对于分类问题，我们采用的评估指标是卡方，假设我们要挑选出3个对于模型最佳性能而言的特征变量，因此我们将K设置成3
select = SelectKBest(score_func=chi2, k=3)
# 拟合数据
z = select.fit_transform(x,y)
filter_1 = select.get_support()
features = np.array(iris_data.feature_names) #此行原代码一直报错，有修改
print("所有的特征: ", features)
print("筛选出来最优的特征是: ", features[filter_1])


#那么对于回归的问题而言，选择上面波士顿房价的例子，同理我们想要筛选出对于模型最佳的性能而言的7个特征变量，评估指标用的是f_regression
boston_data = load_boston()
x = boston_data.data
y = boston_data.target

#然后我们将拟合数据，并且进行特征变量的筛选
select_regression = SelectKBest(score_func=f_regression, k=7)
z = select_regression.fit_transform(x, y)

filter_2 = select_regression.get_support()
features_regression = np.array(boston_data.feature_names)

print("所有的特征变量有:")
print(features_regression)

print("筛选出来的7个特征变量则是:")
print(features_regression[filter_2])