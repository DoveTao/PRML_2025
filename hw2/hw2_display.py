import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from skimage.measure import marching_cubes
from sklearn.metrics import accuracy_score

# 定义生成 3D 月牙数据的函数
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2*np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2*t)
    # 两类数据：一类使用正值，另一类取负值，并在 y 轴上做偏移
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
# 在此生成数据（用于训练和边界绘制）
X, labels = make_moons_3d(n_samples=500, noise=0.2)

# ---------------------------
# 2. 模型训练
# ---------------------------
# 注意：为便于提取概率值，SVM 模型设置 probability=True

# (1) 决策树
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X, labels)

# (2) AdaBoost + 决策树（此处基分类器深度设置为3）
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
    n_estimators=50,
    random_state=42
)
ada_clf.fit(X, labels)

# (3) SVM - 线性核
svm_linear = SVC(kernel='linear', probability=True, random_state=42)
svm_linear.fit(X, labels)

# (4) SVM - 多项式核（degree=3）
svm_poly = SVC(kernel='poly', degree=3, probability=True, random_state=42)
svm_poly.fit(X, labels)

# (5) SVM - RBF 核
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
svm_rbf.fit(X, labels)

# ---------------------------
# 3. 定义 3D 决策边界绘制函数（仅绘制边界）
# ---------------------------
def plot_boundary_3d(clf, ax, resolution=0.2, threshold=0.5, title=""):
    """
    在 3D 空间中仅绘制分类器的决策边界面。
    利用 clf.predict_proba 得到正类概率，然后利用 marching_cubes 提取阈值为 threshold 时的等值面。

    Parameters:
      clf: 分类器，要求具有 predict_proba 或 decision_function 方法
      ax: 3D 子图的 Axes 对象
      resolution: 网格采样的步长
      threshold: 等值面提取的阈值，默认 0.5（即正负类别的分界）
      title: 图形标题
    """
    # 利用训练数据确定空间范围（这里取 X 全局范围）
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    xs = np.arange(x_min, x_max, resolution)
    ys = np.arange(y_min, y_max, resolution)
    zs = np.arange(z_min, z_max, resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # 获取正类概率
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(grid_points)[:, 1]
    elif hasattr(clf, "decision_function"):
        decision = clf.decision_function(grid_points)
        prob = 1/(1+np.exp(-decision))
    else:
        raise ValueError("Classifier does not support probability or decision_function.")
    prob_volume = prob.reshape(xx.shape)

    # 使用 marching_cubes 提取边界等值面，level=threshold
    verts, faces, normals, values = marching_cubes(prob_volume, level=threshold, spacing=(resolution, resolution, resolution))
    # 将体素坐标转换为原始空间坐标
    verts[:, 0] += x_min
    verts[:, 1] += y_min
    verts[:, 2] += z_min

    # 绘制提取的决策边界面
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    cmap="Spectral", alpha=0.7, edgecolor='none')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

# ---------------------------
# 4. 绘制各分类器的 3D 决策边界
# ---------------------------
fig = plt.figure(figsize=(20, 16))

ax1 = fig.add_subplot(231, projection='3d')
plot_boundary_3d(dt_clf, ax1, resolution=0.2, threshold=0.5, title="Decision Tree Boundary")

ax2 = fig.add_subplot(232, projection='3d')
plot_boundary_3d(ada_clf, ax2, resolution=0.2, threshold=0.5, title="AdaBoost + Decision Tree Boundary")

ax3 = fig.add_subplot(233, projection='3d')
plot_boundary_3d(svm_linear, ax3, resolution=0.2, threshold=0.5, title="SVM (Linear Kernel) Boundary")

ax4 = fig.add_subplot(234, projection='3d')
plot_boundary_3d(svm_poly, ax4, resolution=0.2, threshold=0.5, title="SVM (Poly Kernel, degree=3) Boundary")

ax5 = fig.add_subplot(235, projection='3d')
plot_boundary_3d(svm_rbf, ax5, resolution=0.2, threshold=0.5, title="SVM (RBF Kernel) Boundary")

plt.tight_layout()
plt.show()
