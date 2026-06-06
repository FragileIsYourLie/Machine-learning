from sklearn import datasets# 导入数据集模块
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 载入数据集
digits = datasets.load_digits()

# 绘制数据集前 5 个手写数字的灰度图
for index, image in enumerate(digits.images[:5]):
    plt.subplot(2, 5, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

print(digits.target[:5])
train_feature,test_feature,train_target,test_target = train_test_split(digits.data,digits.target,test_size=0.3,random_state=56)
model = SVC()
model.fit(train_feature,train_target)
preds = model.predict(test_feature)
print(accuracy_score(test_target,preds))
print(model.get_params())
print(model.n_support_)          # 每个类别分别有多少支持向量
print(model.support_vectors_.shape)  # 总共有多少个 (总数量, 64)