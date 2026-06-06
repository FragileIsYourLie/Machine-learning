from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# 1. 加载数据
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=56
)

# 2. 训练 RBF SVM（你现在用的模型）
model = SVC()  # 默认 RBF 核
model.fit(X_train, y_train)

# ======================
# 关键：查看支持向量
# ======================
print("每个数字的支持向量数量：", model.n_support_)
print("总支持向量数量：", model.support_vectors_.shape[0])
print("训练集总样本：", len(X_train))

# 3. 画出前 20 个支持向量
plt.figure(figsize=(12, 6))
for i in range(20):
    # 取出第 i 个支持向量
    sv = model.support_vectors_[i]
    # 变回 8x8 图片
    sv_image = sv.reshape(8, 8)

    plt.subplot(2, 10, i + 1)
    plt.imshow(sv_image, cmap=plt.cm.gray_r)
    plt.axis('off')

plt.suptitle("RBF SVM (key samples)", fontsize=16)
plt.show()

plt.figure()
plt.imshow(X_test[1].reshape(8,8), cmap=plt.cm.gray_r)
plt.show()
print(model.predict([X_test[1]]))

print(model.get_params())