import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 更改当前工作目录为脚本所在目录
train_path = "train.csv"
test_path = "test.csv"
os.makedirs("output", exist_ok=True)


def Pre_process(path):
    """读取数据，返回原始特征 X 和标签 y（不做归一化）。"""
    data = pd.read_csv(path, index_col=0)
    X = data.drop("Class", axis=1).values
    y = data["Class"].values
    return X, y


class BpNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BpNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


scaler = StandardScaler()
X_train, y_train = Pre_process(train_path)
X_test, y_test = Pre_process(test_path)

# 归一化：只在训练集上 fit（学习均值/方差），测试集仅 transform（沿用训练集统计量）
# 避免测试集的统计量泄漏进评估，保证指标反映真实泛化能力
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

seed = 233
epochs = 100
torch.manual_seed(seed)
model = BpNet(input_dim=X_train.shape[1], hidden_dim=30, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("Epoch: {}/{}, Loss: {:.4f}".format(epoch + 1, epochs, loss.item()))

model.eval()
with torch.no_grad():
    output = model(X_test)
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y_test.view_as(pred)).sum().item()
    print("Accuracy: {:.2f}%".format(correct / len(y_test) * 100))

# AUC
y_score = output[:, 1].numpy()
precision, recall, _ = precision_recall_curve(y_test.numpy(), y_score)
area = auc(recall, precision)
print("AUC: {:.4f}".format(area))

# plot auc
plt.plot(recall, precision, label="AUC={:.4f}".format(area), color="b")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
# plt.show()
plt.savefig("output/auc.png")
