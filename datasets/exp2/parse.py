import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 中文
plt.rcParams["axes.unicode_minus"] = False  # 负号
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 更改当前工作目录为脚本所在目录

train_path = "data/train.txt"
test_path = "data/test.txt"


def train(data):
    mean = data.groupby("Label").mean()
    m_1 = mean.loc[1]
    m_2 = mean.loc[2]
    point = (m_1 + m_2) / 2
    slope = m_1 - m_2
    return point, slope


def test(data, point, slope):
    data["Predict"] = np.where(
        (data.X - point.X) * slope.X + (data.Y - point.Y) * slope.Y > 0, 1, 2
    )
    data.to_csv("output/test_predict.csv")
    return data


def plot(data, point, slope, filename):
    plt.axline(
        point,
        slope=-slope.X / slope.Y,
        color="green",
        linewidth=2,
        linestyle="--",
        label="Perpendicular Bisector Classifier",
    )
    color = ["r", "b"]
    for _, i in data.iterrows():
        x, y, c = i.iloc[0], i.iloc[1], color[int(i.iloc[2]) - 1]
        plt.scatter(x, y, color=c)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


def evaluate(data):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for _, i in data.iterrows():
        if i.iloc[2] == i.iloc[3]:
            if i.iloc[2] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if i.iloc[2] == 1:
                FP += 1
            else:
                FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("F1 score:", F1)


def main():
    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)
    point, slope = train(train_data)
    # 绘制测试集与决策界
    plot(train_data, point, slope, "./output/train.png")
    # 绘制训练集与决策界
    plot(test_data, point, slope, "./output/test.png")
    test_data = test(test_data, point, slope)
    # 评估测试结果
    evaluate(test_data)


if __name__ == "__main__":
    main()
