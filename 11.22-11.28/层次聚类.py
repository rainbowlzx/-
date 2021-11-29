import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc  #层次聚类的包
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv('data.csv') # dtpye: DataFrame

#对数据进行一个标准化，为了让所有数据在同一个维度便于计算
data_scaled = normalize(data) # dtype: ndarry
data_scaled = pd.DataFrame(data_scaled, columns=data.columns) #转换格式。数据时data_scaled，列的标题是data.columns

#实现层次聚类
plt.figure(figsize=(10, 7)) #figsize:指定figure的宽和高，单位为英寸；
plt.title("Dendrograms")
#linkage实现层次聚类，method选择计算距离的方法；dendrogram用树状图表示结果
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()

#设定阈值为6，切割树状图
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=6, color='r', linestyle='--')
plt.show()

#对2个簇应用层次聚类（前文用不同的函数实现了层次聚类，是为了画树状图，选择阈值，来决定簇的类数）
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_)
plt.show()