import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 配置 matplotlib 使用支持中文的字体（可选）
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 如系统中有SimHei字体
matplotlib.rcParams['axes.unicode_minus'] = False


# 添加截距项：在数据前添加一列常数1
def add_intercept(X):
    return np.column_stack((np.ones(len(X)), X))


# 梯度下降法求解线性回归参数
def gradient_descent(X, y, lr=0.01, num_iter=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for i in range(num_iter):
        # 计算梯度： grad = (1/m) * X^T*(X*theta - y)
        grad = (1 / m) * (X.T @ (X @ theta - y))
        theta = theta - lr * grad
    return theta


# 计算均方误差 (MSE)
def compute_mse(X, y, theta):
    predictions = X @ theta
    mse = np.mean((predictions - y) ** 2)
    return mse


# 绘制拟合结果：同时显示训练数据和测试数据点以及拟合直线
def plot_fit(train_x, train_y, test_x, test_y, theta):
    plt.figure(figsize=(8, 5))
    # 绘制训练集数据点
    plt.scatter(train_x, train_y, label="训练数据", color='blue', marker='o')
    # 绘制测试集数据点
    plt.scatter(test_x, test_y, label="测试数据", color='green', marker='x')

    # 生成连续的 x 值用于绘制拟合直线，取所有数据的最小值到最大值
    x_all = np.concatenate((train_x, test_x))
    x_line = np.linspace(np.min(x_all), np.max(x_all), 100)
    X_line = add_intercept(x_line)
    y_line = X_line @ theta
    plt.plot(x_line, y_line, color='red', label="拟合曲线")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("基于梯度下降法的拟合结果")
    plt.legend()
    plt.show()


def main():
    # 读取 Excel 文件，假设第一个表单为训练数据，第二个表单为测试数据
    data = pd.read_excel("Data4Regression.xlsx", sheet_name=None)
    sheet_names = list(data.keys())
    train_data = data[sheet_names[0]]
    test_data = data[sheet_names[1]]

    # 假设数据的第一列为 x，第二列为 y
    train_x = train_data.iloc[:, 0].values
    train_y = train_data.iloc[:, 1].values
    test_x = test_data.iloc[:, 0].values
    test_y = test_data.iloc[:, 1].values

    # 构建设计矩阵：添加截距项
    X_train = add_intercept(train_x)
    X_test = add_intercept(test_x)

    # 使用梯度下降法拟合模型
    theta = gradient_descent(X_train, train_y, lr=0.01, num_iter=1000)

    # 分别计算训练集和测试集的均方误差
    train_mse = compute_mse(X_train, train_y, theta)
    test_mse = compute_mse(X_test, test_y, theta)

    print("拟合参数 theta:", theta)
    print("训练集均方误差:", train_mse)
    print("测试集均方误差:", test_mse)

    # 绘制训练集与测试集数据点和拟合直线
    plot_fit(train_x, train_y, test_x, test_y, theta)


if __name__ == '__main__':
    main()
