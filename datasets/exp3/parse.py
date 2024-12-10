import os
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

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
for k in range(0, len(keywords) + 1):
    train = Pre_Process(train_path, keywords[:k])
    test = Pre_Process(test_path, keywords[:k])
    # 训练模型
    model = BernoulliNB()
    model.fit(train.drop("Crime type", axis=1), train["Crime type"])

    # 预测结果
    pred = model.predict(test.drop("Crime type", axis=1))
    acc.append(
        accuracy_score(test["Crime type"], pred),
    )

# plot
plt.plot(range(0, len(keywords) + 1), acc, color="b")
plt.xlabel("Number of Keywords")
plt.ylabel("Accuracy")
points = [(0, acc[0]), (5, acc[5]), (len(acc), acc[len(acc) - 1])]
xticks = [x for x, _ in points]
yticks = [y for _, y in points]
plt.xticks(xticks, [f"{x}" for x in xticks])
plt.yticks(yticks, [f"{y * 100:.2f}%" for y in yticks])
plt.plot([points[1][0], points[1][0]], [0.90, points[1][1]], color="r", linestyle="--")
plt.plot([0.00, points[1][0]], [points[1][1], points[1][1]], color="r", linestyle="--")
plt.xlim(0, len(keywords) + 1)
plt.ylim(0.90, 1.01)
# plt.show()
plt.savefig("accuracy.png", bbox_inches="tight")
