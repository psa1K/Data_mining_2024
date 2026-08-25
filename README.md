# Data Mining 2024 大作业

数据挖掘课程（2024）大作业仓库。包含课程各周实验的**数据集、Python 实现脚本**，以及**期末大作业**的六个子任务、报告 LaTeX 模板与课件。

## 目录结构

```
Data_mining_2024/
├── datasets/            # 数据集 + 实验/期末代码
│   ├── exp1/ ~ exp5/    # 五次课程实验（各含 expN.md 题目说明 + parse.py 实现 + 数据）
│   └── final/           # 期末大作业（六个子任务）
├── demos/               # 报告 LaTeX 模板（report.tex）
└── slides/              # 课程课件 PDF
```

## 课程实验（exp1 ~ exp5）

每个实验目录包含题目说明 `expN.md`、解决方案 `parse.py`（pandas / scikit-learn / matplotlib 实现）以及对应数据集。

| 目录 | 实验内容 | 方法 |
|------|---------|------|
| `exp1` | 图像纹理特征提取 | 灰度共生矩阵 / 局部二值模式 |
| `exp2` | 垂直平分分类器 | 基于训练数据训练，对测试集分类 |
| `exp3` | 犯罪类型预测 | 朴素贝叶斯 |
| `exp4` | 犯罪类型预测 | 决策树 |
| `exp5` | 信用卡欺诈检测 | BP 神经网络 / 感知机 |

## 期末大作业（final）

`datasets/final/` 下包含六个子任务，各提供测试集，训练集需自行选择：

| 子任务 | 数据形态 |
|--------|---------|
| 区域分割 | 图像（.jpg） |
| 台风预测 | 说明 + 数据 |
| 模型对比 | train.csv / test.csv |
| 特征选择 | training_data.csv |
| 福字识别 | 图像（.png） |
| 飞机检测 | train / test + parse.py / parse.ipynb |

## 运行方式

```bash
# 进入对应实验目录后直接运行
python parse.py
```

脚本内部会 `os.chdir` 切换到自身所在目录，请在**各自目录下**执行。

## 报告

使用 `demos/report.tex` 的 LaTeX 模板撰写实验与期末报告。

## 备注

- 仓库包含较大数据集（约 160 MB），请勿随意提交生成的输出文件（`output/`、图片、中间产物）。
- 题目说明与课件为中文，保持原文。
