import pandas as pd  # 导入数据处理模块
from matplotlib import pyplot as plt
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score  # 导入轮廓系数计算模块

df = pd.read_csv("http://labfile.oss.aliyuncs.com/courses/866/cluster_data.csv", header=0)  # 导入数据文件
print(df.head())
X = df['x']  # 定义横坐标数据
y = df['y']  # 定义纵坐标数据
plt.scatter(X, y)  # 绘制散点图
model = k_means(df,n_clusters=3)
print(model)
cluster_centers = model[0]  # 聚类中心数组
cluster_labels = model[1]  # 聚类标签数组
plt.scatter(X, y, c=cluster_labels)  # 绘制样本并按聚类标签标注颜色
# 绘制聚类中心点，标记成五角星样式，以及红色边框
for center in cluster_centers:
    plt.scatter(center[0], center[1], marker="*", edgecolors="black")
plt.show()
plt.figure()
index = []  # 横坐标数组
inertia = []  # 纵坐标数组

# K 从 1~ 10 聚类
for i in range(9):
    model = k_means(df, n_clusters=i + 1)
    index.append(i + 1)
    inertia.append(model[2])

# 绘制折线图
plt.plot(index, inertia, "-o")

plt.figure()
index2 = []  # 横坐标
silhouette = []  # 轮廓系数列表

# K 从 2 ~ 10 聚类
for i in range(8):
    model = k_means(df, n_clusters=i + 2)
    index2.append(i + 2)
    silhouette.append(silhouette_score(df, model[1]))

print(silhouette)  # 输出不同聚类下的轮廓系数

# 绘制折线图
plt.plot(index2, silhouette, "-o")

plt.show()