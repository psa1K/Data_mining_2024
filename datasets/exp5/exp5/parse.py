import os
import pandas as pd


os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 更改当前工作目录为脚本所在目录
train_path = "train.csv"
test_path = "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(train["Class"].value_counts())
print(test["Class"].value_counts())
