from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# ===================== 第一部分：线性回归 =============================
diabetes = datasets.load_diabetes()
diabetes_feature = diabetes.data[:, np.newaxis, 2]
diabetes_target = diabetes.target

train_feature, test_feature, train_target, test_target = train_test_split(
    diabetes_feature, diabetes_target, test_size=0.3, random_state=56)

model = LinearRegression()
model.fit(train_feature, train_target)

plt.scatter(train_feature, train_target, color='black')
plt.scatter(test_feature, test_target, color='red')
plt.plot(test_feature, model.predict(test_feature), color='blue', linewidth=3)

plt.legend(('Fit line', 'Train Set', 'Test Set'), loc='lower right')
plt.title('LinearRegression Example')

# ===================== 第二部分：感知机 + 画分界线 + 准确率 =========================
# 生成分类数据
X, y = datasets.make_classification(n_features=2, n_redundant=0,
                                    n_informative=1, n_clusters_per_class=1, random_state=1)

# 切分数据
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=56)

# 训练感知机
model = Perceptron()
model.fit(train_x, train_y)

# 预测
preds = model.predict(test_x)

# ===================== 【打印准确率】 =====================
print("感知机模型准确率：", accuracy_score(test_y, preds))

# ===================== 【画感知机分界线】 =====================
plt.figure()  # 新建画布
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")

# 获取感知机的权重 w 和偏置 b
w = model.coef_[0]
b = model.intercept_[0]

# 构建分界线：w[0]*x + w[1]*y + b = 0 → y = -(w[0]/w[1])x - b/w[1]
x_points = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 100)
y_points = -(w[0] / w[1]) * x_points - b / w[1]

# 画分界线
plt.plot(x_points, y_points, 'k-', lw=2, c='y')
plt.title("Perceptron 感知机分界线")

plt.show()