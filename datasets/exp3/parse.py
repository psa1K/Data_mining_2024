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


def build_encoder(df):
    """在训练集上学习类别编码，返回共享的标签编码器与 one-hot 规范列名。"""
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df["Category"])

    hour_cols = pd.get_dummies(pd.to_datetime(df["Dates"]).dt.hour).columns
    day_cols = pd.get_dummies(df["DayOfWeek"]).columns
    district_cols = pd.get_dummies(df["PdDistrict"]).columns
    return encoder, hour_cols, day_cols, district_cols


def Pre_Process(df, keywords, encoder, hour_cols, day_cols, district_cols):
    # 标签编码：用训练集拟合好的编码器 transform（不再重新 fit，保证映射一致）
    crime_type_encode = encoder.transform(df["Category"])

    # one-hot 编码后按训练集列结构对齐（test 缺失的类别补 0，多余的丢弃）
    hour = pd.get_dummies(pd.to_datetime(df["Dates"]).dt.hour)
    hour = hour.reindex(columns=hour_cols, fill_value=0)
    day = pd.get_dummies(df["DayOfWeek"]).reindex(columns=day_cols, fill_value=0)
    police_district = pd.get_dummies(df["PdDistrict"]).reindex(
        columns=district_cols, fill_value=0
    )

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


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
keywords = keyword_extract()
encoder, hour_cols, day_cols, district_cols = build_encoder(train_df)

acc = []
for k in range(0, len(keywords) + 1):
    train = Pre_Process(
        train_df, keywords[:k], encoder, hour_cols, day_cols, district_cols
    )
    test = Pre_Process(
        test_df, keywords[:k], encoder, hour_cols, day_cols, district_cols
    )
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
