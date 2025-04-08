import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 定义生成 3D 月牙数据的函数
def make_moons_3d(n_samples=500, noise=0.1):
    # 生成原始的 2D 月牙数据（每一类样本数为 n_samples）
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # 在第三个维度添加正弦变化

    # 两类数据的拼接：一类使用正值，一类取负值，并在 y 轴上做偏移
    X = np.vstack([
        np.column_stack([x, y, z]),
        np.column_stack([-x, y - 1, -z])
    ])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # 添加高斯噪声
    X += np.random.normal(scale=noise, size=X.shape)
    return X, labels

# ---------------------------
# 1. 数据生成
# ---------------------------
# 训练数据：1000 个样本（500 个 C0，500 个 C1）
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)
# 测试数据：500 个样本（250 个 C0，250 个 C1）
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

# 可视化训练数据的 3D 散点图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Training Data 3D Make Moons')
plt.show()

# ---------------------------
# 2. 模型训练与预测
# ---------------------------

# (1) 决策树模型
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# (2) AdaBoost + 决策树（默认基分类器深度设为 1 作为基准）
ada_clf_default = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    random_state=42
)
ada_clf_default.fit(X_train, y_train)
y_pred_ada = ada_clf_default.predict(X_test)
acc_ada = accuracy_score(y_test, y_pred_ada)

# (3) 支持向量机 (SVM)

# SVM - 线性核
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)
acc_svm_linear = accuracy_score(y_test, y_pred_svm_linear)

# SVM - 多项式核（degree=3）
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_poly.fit(X_train, y_train)
y_pred_svm_poly = svm_poly.predict(X_test)
acc_svm_poly = accuracy_score(y_test, y_pred_svm_poly)

# SVM - RBF 核
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test)
acc_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)

# 输出测试集分类准确率
print("测试集分类准确率：")
print("决策树         : {:.2f}%".format(acc_dt * 100))
print("AdaBoost       : {:.2f}%".format(acc_ada * 100))
print("SVM (线性核)    : {:.2f}%".format(acc_svm_linear * 100))
print("SVM (多项式核)  : {:.2f}%".format(acc_svm_poly * 100))
print("SVM (RBF核)    : {:.2f}%".format(acc_svm_rbf * 100))

# ---------------------------
# 3. 定义 3D 决策区域绘图函数
# ---------------------------
def plot_decision_regions_3d(clf, X, y, ax, resolution=0.8, title=""):
    # 确定数据范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    # 构造 3D 网格
    xx, yy, zz = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
        np.arange(z_min, z_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    # 对网格点做出预测
    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制网格点的预测结果（点较小且透明度较低）
    ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
               c=Z, cmap=plt.cm.RdBu, alpha=0.03, marker='.', s=1)

    # 叠加原始训练数据点
    ax.scatter(X[y==0, 0], X[y==0, 1], X[y==0, 2],
               c='blue', edgecolor='k', marker='o', s=30, label='C0')
    ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2],
               c='red', edgecolor='k', marker='^', s=30, label='C1')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

# 绘制各分类器的 3D 决策区域图
fig = plt.figure(figsize=(20, 16))
ax1 = fig.add_subplot(231, projection='3d')
plot_decision_regions_3d(dt_clf, X_train, y_train, ax1, resolution=0.8, title="Decision Tree")

ax2 = fig.add_subplot(232, projection='3d')
plot_decision_regions_3d(ada_clf_default, X_train, y_train, ax2, resolution=0.8, title="AdaBoost (默认基分类器深度 = 1)")

ax3 = fig.add_subplot(233, projection='3d')
plot_decision_regions_3d(svm_linear, X_train, y_train, ax3, resolution=0.8, title="SVM (Linear Kernel)")

ax4 = fig.add_subplot(234, projection='3d')
plot_decision_regions_3d(svm_poly, X_train, y_train, ax4, resolution=0.8, title="SVM (Poly Kernel)")

ax5 = fig.add_subplot(235, projection='3d')
plot_decision_regions_3d(svm_rbf, X_train, y_train, ax5, resolution=0.8, title="SVM (RBF Kernel)")
plt.tight_layout()
plt.show()

# ---------------------------
# 4. AdaBoost + DecisionTree 随基分类器深度变化的实验
# ---------------------------
depths = range(1, 11)  # 考察决策树深度从 1 到 10
accuracy_list = []

for depth in depths:
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth, random_state=42),
        n_estimators=50,
        random_state=42
    )
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)
    print(f"AdaBoost with DecisionTree max_depth = {depth}: accuracy = {acc * 100:.2f}%")

# 绘制随决策树深度变化的准确率折线图
plt.figure(figsize=(8, 6))
plt.plot(depths, [acc * 100 for acc in accuracy_list], marker='o', linestyle='-')
plt.xlabel("Decision Tree max_depth")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Decision Tree Depth in AdaBoost")
plt.grid(True)
plt.show()
