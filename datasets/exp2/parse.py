import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 中文
plt.rcParams["axes.unicode_minus"] = False  # 负号
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 更改当前工作目录为脚本所在目录

train_path = "data/train.txt"
test_path = "data/test.txt"
output_dir = "output"


def train(data):
    """学习垂直平分线分类器参数，返回类别标签、中点与法向量。"""
    means = data.groupby("Label").mean()
    labels = sorted(means.index.tolist())  # 动态取类别，不假设是 1/2
    m_1, m_2 = means.loc[labels[0]], means.loc[labels[1]]
    midpoint = (m_1 + m_2) / 2   # 垂直平分线经过的中点
    normal = m_1 - m_2           # 两簇中心差向量 = 决策线的法向量
    return labels, midpoint, normal


def predict(data, labels, midpoint, normal):
    """用中点 + 法向量判定每个样本落在决策线的哪一侧。"""
    data["Predict"] = np.where(
        (data.X - midpoint.X) * normal.X + (data.Y - midpoint.Y) * normal.Y > 0,
        labels[0],
        labels[1],
    )
    data.to_csv(os.path.join(output_dir, "test_predict.csv"))
    return data


def plot(data, labels, midpoint, normal, filename):
    plt.axline(
        midpoint,
        slope=-normal.X / normal.Y,
        color="green",
        linewidth=2,
        linestyle="--",
        label="Perpendicular Bisector Classifier",
    )
    color_map = {labels[0]: "r", labels[1]: "b"}
    for _, i in data.iterrows():
        x, y, label = i.iloc[0], i.iloc[1], i.iloc[2]
        plt.scatter(x, y, color=color_map[label])
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate(data, positive_label):
    """计算分类指标，防止分母为 0 时崩溃。"""
    TP = TN = FP = FN = 0
    for _, i in data.iterrows():
        actual, pred = i.iloc[2], i.iloc[3]
        if actual == pred:
            if actual == positive_label:
                TP += 1
            else:
                TN += 1
        else:
            if actual == positive_label:
                FP += 1
            else:
                FN += 1
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"accuracy:  {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")


def main():
    os.makedirs(output_dir, exist_ok=True)
    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)
    labels, midpoint, normal = train(train_data)
    # 绘制训练集与决策界
    plot(train_data, labels, midpoint, normal, os.path.join(output_dir, "train.png"))
    # 绘制测试集与决策界
    plot(test_data, labels, midpoint, normal, os.path.join(output_dir, "test.png"))
    test_data = predict(test_data, labels, midpoint, normal)
    # 评估测试结果
    evaluate(test_data, labels[0])


if __name__ == "__main__":
    main()
