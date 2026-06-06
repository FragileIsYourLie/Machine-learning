import pandas as pd  # 加载 pandas 模块
from matplotlib import pyplot as plt  # 加载绘图模块
# 集成方法分类器
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
# 高斯过程分类器
from sklearn.gaussian_process import GaussianProcessClassifier
# 广义线性分类器
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
# K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
# 神经网络分类器
from sklearn.neural_network import MLPClassifier
# 决策树分类器
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
# 支持向量机分类器
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split  # 导入数据集切分模块
from sklearn.metrics import accuracy_score  # 导入准确度评估模块
from matplotlib.colors import ListedColormap  # 加载色彩模块
import numpy as np  # 导入数值计算模块
from matplotlib.colors import ListedColormap  # 加载色彩模块
import numpy as np  # 导入数值计算模块
from tqdm import tqdm

# 读取 csv 文件, 并将第一行设为表头
data = pd.read_csv(
    "https://labfile.oss.aliyuncs.com/courses/866/class_data.csv", header=0)

print(data.head()) # 输出数据预览
plt.scatter(data["X"], data['Y'], c=data['CLASS'])

# 建立模型
models = [
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    GaussianProcessClassifier(),
    PassiveAggressiveClassifier(),
    RidgeClassifier(),
    SGDClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    MLPClassifier(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    SVC(),
    LinearSVC()
]

# 依次为模型命名
classifier_Names = ['AdaBoost',
                    'Bagging',
                    'ExtraTrees',
                    'GradientBoosting',
                    'RandomForest',
                    'GaussianProcess',
                    'PassiveAggressive',
                    'Ridge',
                    'SGD',
                    'KNeighbors',
                    'GaussianNB',
                    'MLP',
                    'DecisionTree',
                    'ExtraTree',
                    'SVC',
                    'LinearSVC'
                    ]
X_train, X_test, y_train, y_test = train_test_split(
    data[['X','Y']], data['CLASS'], test_size=.3)  # 切分数据集

# 遍历所有模型
for name, model in zip(classifier_Names, models):
    model.fit(X_train, y_train)  # 训练模型
    pre_labels = model.predict(X_test)  # 模型预测
    score = accuracy_score(y_test, pre_labels)  # 计算预测准确度
    print('%s: %.2f' % (name, score))  # 输出模型准确度

# 绘制数据集
i = 1
cm = plt.cm.Blues
cm_color = ListedColormap(['red', 'yellow'])
# 栅格化
x_min, x_max = data['X'].min() - .5, data['X'].max() + .5
y_min, y_max = data['Y'].min() - .5, data['Y'].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                     np.arange(y_min, y_max, .1))
# 模型迭代
plt.figure(figsize=(20, 10))

for name, model in tqdm(list(zip(classifier_Names, models))):
    ax = plt.subplot(4, 4, i)

    model.fit(X_train, y_train)
    pre_labels = model.predict(X_test)
    score = accuracy_score(y_test, pre_labels)

    # 决策边界
    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.6)

    # 绘制点
    ax.scatter(X_train['X'], X_train['Y'], c=y_train, cmap=cm_color)
    ax.scatter(X_test['X'], X_test['Y'], c=y_test,
               cmap=cm_color, edgecolors='black')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('%s | %.2f' % (name, score))
    i += 1

plt.tight_layout()
plt.show()
