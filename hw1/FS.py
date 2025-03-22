import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 配置 matplotlib 使用支持中文的字体（可选）
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 如系统中有SimHei字体
matplotlib.rcParams['axes.unicode_minus'] = False


# 生成傅里叶基函数特征
def fourier_features(x, order, period):
    """
    输入：
      x：一维数组，数据自变量
      order：傅里叶级数的阶数（即包含几个谐波）
      period：周期，通常可以设置为 max(x)-min(x) 或者根据实际数据指定
    输出：
      返回形状为 (len(x), 1+2*order) 的设计矩阵，其中第一列为常数项，
      随后依次为 cos(2πk x/period) 和 sin(2πk x/period)（k=1,...,order）
    """
    N = len(x)
    features = [np.ones_like(x)]
    for k in range(1, order + 1):
        features.append(np.cos(2 * np.pi * k * x / period))
        features.append(np.sin(2 * np.pi * k * x / period))
    return np.column_stack(features)


# 利用正规方程求解最小二乘法问题
def least_squares_fit(X, y):
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return theta


# 计算均方误差（MSE）
def compute_mse(predictions, y):
    mse = np.mean((predictions - y) ** 2)
    return mse


# 绘制数据与拟合曲线：同时显示训练数据点和测试数据点
def plot_fit(train_x, train_y, test_x, test_y, theta, order, period, title="傅里叶序列拟合结果"):
    # 合并训练和测试数据以确定 x 轴范围
    x_all = np.concatenate((train_x, test_x))
    x_line = np.linspace(np.min(x_all), np.max(x_all), 1000)
    X_line = fourier_features(x_line, order, period)
    y_line = X_line @ theta

    plt.figure(figsize=(8, 5))
    # 绘制训练数据点
    plt.scatter(train_x, train_y, label="训练数据", color='blue', marker='o')
    # 绘制测试数据点
    plt.scatter(test_x, test_y, label="测试数据", color='green', marker='x')
    # 绘制拟合曲线
    plt.plot(x_line, y_line, label="拟合曲线", color='red')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    # 读取 Excel 数据，假设第一个表单为训练数据，第二个表单为测试数据
    data = pd.read_excel("Data4Regression.xlsx", sheet_name=None)
    sheet_names = list(data.keys())
    train_data = data[sheet_names[0]]
    test_data = data[sheet_names[1]]

    # 提取训练和测试数据，假设第一列为 x，第二列为 y
    train_x = train_data.iloc[:, 0].values
    train_y = train_data.iloc[:, 1].values
    test_x = test_data.iloc[:, 0].values
    test_y = test_data.iloc[:, 1].values

    # 设置傅里叶级数的阶数（谐波个数）
    order = 5  # 可根据数据复杂度调整
    # 设定周期：这里简单取训练集 x 的范围作为周期
    period = np.max(train_x) - np.min(train_x)

    # 构建设计矩阵（傅里叶基函数特征）
    X_train = fourier_features(train_x, order, period)
    X_test = fourier_features(test_x, order, period)

    # 求解最小二乘法参数 theta
    theta = least_squares_fit(X_train, train_y)

    # 计算训练集和测试集的预测值
    train_pred = X_train @ theta
    test_pred = X_test @ theta

    # 计算均方误差
    train_mse = compute_mse(train_pred, train_y)
    test_mse = compute_mse(test_pred, test_y)

    print("拟合系数 theta:")
    print(theta)
    print("训练集均方误差:", train_mse)
    print("测试集均方误差:", test_mse)

    # 绘制训练集和测试集的拟合结果在同一图中
    plot_fit(train_x, train_y, test_x, test_y, theta, order, period, title="傅里叶序列拟合（训练+测试）")


if __name__ == '__main__':
    main()
