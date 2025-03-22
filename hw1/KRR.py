import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

# 配置 matplotlib 使用支持中文的字体（可选）
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 如系统中有 SimHei 字体
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_results(train_x, train_y, test_x, test_y, model, title):
    """
    绘制拟合结果图：
    - 显示训练数据（蓝色圆点）和测试数据（绿色叉号）
    - 绘制模型预测得到的连续拟合曲线（红色线）
    """
    # 合并训练和测试数据的 x 值，确定 x 轴范围
    x_all = np.concatenate((train_x, test_x))
    x_line = np.linspace(np.min(x_all), np.max(x_all), 1000)
    y_line = model.predict(x_line.reshape(-1, 1))

    plt.figure(figsize=(8, 5))
    # 绘制训练数据点
    plt.scatter(train_x, train_y, color='blue', marker='o', label="训练数据")
    # 绘制测试数据点
    plt.scatter(test_x, test_y, color='green', marker='x', label="测试数据")
    # 绘制拟合曲线
    plt.plot(x_line, y_line, color='red', label="拟合曲线")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    # 读取 Excel 数据，假设 "Data4Regression.xlsx" 中第一个表单为训练数据，第二个表单为测试数据
    data = pd.read_excel("Data4Regression.xlsx", sheet_name=None)
    sheet_names = list(data.keys())
    train_data = data[sheet_names[0]]
    test_data = data[sheet_names[1]]

    # 假设数据的第一列为 x，第二列为 y
    train_x = train_data.iloc[:, 0].values.reshape(-1, 1)
    train_y = train_data.iloc[:, 1].values
    test_x = test_data.iloc[:, 0].values.reshape(-1, 1)
    test_y = test_data.iloc[:, 1].values

    # 构建核岭回归模型，使用高斯核（RBF核）
    # 参数说明：
    #   kernel='rbf' 表示使用高斯核；
    #   gamma 控制高斯核宽度（值越大，核函数越窄）；
    #   alpha 为正则化参数
    model = KernelRidge(kernel='rbf', gamma=0.8, alpha=1.0)
    model.fit(train_x, train_y)

    # 对训练集和测试集进行预测，并计算均方误差（MSE）
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    mse_train = mean_squared_error(train_y, train_pred)
    mse_test = mean_squared_error(test_y, test_pred)

    print("【高斯核核岭回归模型】")
    print("训练集均方误差:", mse_train)
    print("测试集均方误差:", mse_test)

    # 绘制拟合结果图，展示训练数据、测试数据和拟合曲线
    plot_results(train_x, train_y, test_x, test_y, model, "高斯核核岭回归拟合结果")


if __name__ == '__main__':
    main()
