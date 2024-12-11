import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree


os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 更改当前工作目录为脚本所在目录

train_path = "train.csv"
test_path = "test.csv"


def keyword_extract(path="train.csv"):
    df = pd.read_csv(path)
    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
    tfidf = pd.DataFrame(
        vectorizer.fit_transform(df["Descript"]).toarray(),
        columns=vectorizer.get_feature_names_out(),
    )
    keywords = tfidf.idxmax(axis=1)
    keywords = keywords.value_counts(normalize=True).index.str.upper().tolist()
    return keywords


def Pre_Process(path, keywords):
    df = pd.read_csv(path)

    # 将Category进行整数编码
    encoder = preprocessing.LabelEncoder()
    crime_type_encode = encoder.fit_transform(df["Category"])

    # 将时间进行one-hot编码
    hour = pd.to_datetime(df["Dates"]).dt.hour
    hour = pd.get_dummies(hour)
    day = pd.get_dummies(df["DayOfWeek"])

    # 将所属警区进行one-hot编码
    police_district = pd.get_dummies(df["PdDistrict"])

    # 利用 TF-IDF 特征进行编码
    matrix = pd.DataFrame(0, index=df.index, columns=keywords)
    for keyword in keywords:
        matrix[keyword] = df["Descript"].apply(
            lambda x: True if keyword in x else False
        )

    # 将特征合并
    data = pd.concat([hour, day, police_district, matrix], axis=1)
    data["Crime type"] = crime_type_encode

    # Feature names are only supported if all input features have string names
    data.columns = data.columns.astype(str)
    return data


keywords = keyword_extract()
acc = []
depth = []
for k in range(0, len(keywords) + 1):
    train = Pre_Process(train_path, keywords[:k])
    test = Pre_Process(test_path, keywords[:k])
    # 训练模型
    model = DecisionTreeClassifier()
    model.fit(train.drop("Crime type", axis=1), train["Crime type"])
    # 预测结果
    y_pred = model.predict(test.drop("Crime type", axis=1))
    acc.append(accuracy_score(test["Crime type"], y_pred))
    depth.append(model.get_depth())

# plot
fig, ax1 = plt.subplots()
ax1.set_xlabel("Number of Keywords")
ax1.set_ylabel("Accuracy")
ax1.plot(range(0, len(keywords) + 1), acc, color="r", label="Accuracy")
ax1.tick_params(axis="y")
# 创建第二个坐标轴共享x轴
ax2 = ax1.twinx()
ax2.set_ylabel("Depth")
ax2.plot(range(0, len(keywords) + 1), depth, color="b", label="Depth")
ax2.tick_params(axis="y")
# 显示图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left")
# plt.show()
plt.savefig("accuracy+depth.png")
# 绘制决策树

plt.figure(figsize=(10, 10))
plot_tree(
    model,
    filled=True,
    rounded=True,
    class_names=list(pd.read_csv(train_path)["Category"].unique()),
    feature_names=list(train.drop("Crime type", axis=1).columns),
)
plt.savefig("decision_tree.png")
