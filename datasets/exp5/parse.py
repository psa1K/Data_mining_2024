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

train = pd.read_csv(train_path, index_col=0)
test = pd.read_csv(test_path, index_col=0)


def Pre_process(path):
    data = pd.read_csv(path, index_col=0)
    X = data.drop("Class", axis=1).values
    y = data["Class"].values
    X = scaler.fit_transform(X)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
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
plt.savefig("auc.png")
