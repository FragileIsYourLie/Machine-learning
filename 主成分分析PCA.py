from sklearn import datasets  # 导入数据集模块
import matplotlib.pyplot as plt  # 导入绘图模块
from sklearn import decomposition
from sklearn.cluster import KMeans
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 载入数据集
digits_data = datasets.load_digits()

# 绘制数据集前 5 个手写数字的灰度图
for index, image in enumerate(digits_data.images[:5]):
    plt.subplot(1, 5, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

X = digits_data.data
y = digits_data.target
# PCA 将数据降为 2 维
estimator = decomposition.PCA(n_components=2)
reduce_data = estimator.fit_transform(X)

# 建立 K-Means 并输入数据
model = KMeans(n_clusters=10)
model.fit(reduce_data)

# 计算聚类过程中的决策边界
x_min, x_max = reduce_data[:, 0].min() - 1, reduce_data[:, 0].max() + 1
y_min, y_max = reduce_data[:, 1].min() - 1, reduce_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                     np.arange(y_min, y_max, .05))

result = model.predict(np.c_[xx.ravel(), yy.ravel()])

# 将决策边界绘制绘制出来
result = result.reshape(xx.shape)
plt.figure(figsize=(10, 5))
plt.contourf(xx, yy, result, cmap=plt.cm.Greys)
plt.scatter(reduce_data[:, 0], reduce_data[:, 1], c=y, s=15)

# 绘制聚类中心点
center = model.cluster_centers_
plt.scatter(center[:, 0], center[:, 1], marker='p',
            linewidths=2, color='b', edgecolors='w', zorder=20)

# 图像参数设置
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

model = RandomForestClassifier()
cross_val_score(model, X, y, cv=5).mean()  # 5 折交叉验证平均准确度

estimator = decomposition.PCA(n_components=5) # 从 10 个特征缩减为 5 个特征
X_pca = estimator.fit_transform(X)

model = RandomForestClassifier()
a1 = cross_val_score(model, X_pca, y, cv=5).mean()  # 5 折交叉验证平均准确度
print("没有降维时的5-flod交叉验证平均准确度：",a1)

estimator = decomposition.PCA(n_components=5) # 从 10 个特征缩减为 5 个特征
X_pca = estimator.fit_transform(X)

model = RandomForestClassifier()
a2 = cross_val_score(model, X_pca, y, cv=5).mean()  # 5 折交叉验证平均准确度
print("降维后的5-flod交叉验证平均准确度：",a2)

# 本实验给我们的启示是：很多时候降维之后的数据用来训练，准确度不会降低很多