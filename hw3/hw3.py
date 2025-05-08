import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 超参数设置
SEQ_LEN = 24  # 每个序列的时间步长
BATCH_SIZE = 128  # 批量大小
EPOCHS = 50  # 训练周期
LR = 0.001  # 初始学习率
HIDDEN_SIZE = 64  # LSTM 隐藏层大小
NUM_LAYERS = 3  # LSTM 层数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# 修正数据处理，确保特征数量一致
def load_and_scale(file, scaler=None, encoder=None, fit=False):
    df = pd.read_csv(file)
    df = df[['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']].dropna()

    # 提取数值特征
    numeric_features = df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']].values
    wind_dir = df[['wnd_dir']].values  # 保持二维结构

    if fit:
        # 训练时执行标准化和独热编码
        scaler = MinMaxScaler()
        numeric_scaled = scaler.fit_transform(numeric_features)

        # 对风向进行独热编码
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        wind_encoded = encoder.fit_transform(wind_dir)

        # 合并数值特征和风向编码特征（确保总特征数为 11）
        data = np.concatenate([numeric_scaled, wind_encoded], axis=1)
        print(f"Feature shape after concatenation (train): {data.shape}")  # 打印训练数据的特征形状
        return data, scaler, encoder
    else:
        # 使用训练数据的标准化器和编码器对测试数据进行转换
        numeric_scaled = scaler.transform(numeric_features)
        wind_encoded = encoder.transform(wind_dir)

        # 合并数值特征和风向编码特征（确保总特征数为 11）
        data = np.concatenate([numeric_scaled, wind_encoded], axis=1)
        print(f"Feature shape after concatenation (test): {data.shape}")  # 打印测试数据的特征形状
        return data


# 重新检查数据
train_data, scaler, encoder = load_and_scale('LSTM-Multivariate_pollution.csv', fit=True)
test_data = load_and_scale('pollution_test_data1.csv', scaler, encoder=encoder)

# 检查数据的形状，确保是 (样本数, 时间步数, 特征数)
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)


# 构建时间序列数据集
# 修正 PollutionDataset 类中的数据处理
class PollutionDataset(TensorDataset):
    def __init__(self, data, seq_len):
        self.X = []
        self.y = []
        for i in range(seq_len, len(data)):
            # 使用全特征 (包括风向的独热编码)
            self.X.append(data[i - seq_len:i, :])  # 包括所有特征
            self.y.append(data[i, 0])  # 目标列为污染值（PM2.5）

        # 将 X 和 y 转换为张量
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 加载数据
train_data, scaler, encoder = load_and_scale('LSTM-Multivariate_pollution.csv', fit=True)
test_data = load_and_scale('pollution_test_data1.csv', scaler, encoder=encoder)

# # 检查数据的形状，确保特征数量正确
# print(f"Train Data Shape: {train_data.shape}")
# print(f"Test Data Shape: {test_data.shape}")

# 构建数据集
train_dataset = PollutionDataset(train_data, SEQ_LEN)
test_dataset = PollutionDataset(test_data, SEQ_LEN)

# # 检查数据集大小
# print("Train Dataset Length:", len(train_dataset))
# print("Test Dataset Length:", len(test_dataset))

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# 打印数据样本的形状
for X_batch, y_batch in train_loader:
    print(f"X_batch shape: {X_batch.shape}")
    break  # 仅查看第一个批次

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出
        return out


# 模型初始化
input_size = 11  # 特征数量（数值特征 + 风向独热编码）
model = LSTMModel(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1).to(DEVICE)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

# 训练过程
train_losses = []
test_losses = []
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 评估模型
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # 学习率调度
    scheduler.step(avg_test_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# 绘制训练和测试损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Losses')
plt.legend()
plt.show()

# 进行预测
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        output = model(X_batch)
        predictions.append(output.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# 反标准化
predictions_inv = scaler.inverse_transform(np.hstack([predictions, np.zeros((predictions.shape[0], 10))]))[:, 0]
actuals_inv = scaler.inverse_transform(np.hstack([actuals, np.zeros((actuals.shape[0], 10))]))[:, 0]

# 绘制预测结果与真实结果对比
plt.figure(figsize=(10, 5))
plt.plot(actuals_inv, label='True PM2.5')
plt.plot(predictions_inv, label='Predicted PM2.5')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.title('True vs Predicted Pollution Levels')
plt.legend()
plt.show()

# 计算评估指标
mse = mean_squared_error(actuals_inv, predictions_inv)
mae = mean_absolute_error(actuals_inv, predictions_inv)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
