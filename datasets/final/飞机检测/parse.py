import cv2
import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random.seed(233)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def getTrainData(path):
    imgs = []
    labels = []
    for xmlPath in os.listdir(path + "Annotation/xml/"):
        tree = ET.parse(path + "Annotation/xml/" + xmlPath)
        root = tree.getroot()
        # 获取图像尺寸
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        # 获取图像名
        imgPath = root.find("filename").text
        # 读取图像
        img = cv2.imread(path + "JPEGImages/" + imgPath, cv2.IMREAD_GRAYSCALE)
        # 获取边界框
        targetRects = []
        for num, obj in enumerate(root.findall("object")):
            # 获取边界框
            bndbox = obj.find("bndbox")
            xmin = max(0, int(bndbox.find("xmin").text))
            ymin = max(0, int(bndbox.find("ymin").text))
            xmax = max(0, int(bndbox.find("xmax").text))
            ymax = max(0, int(bndbox.find("ymax").text))
            targetRects.append((xmin, ymin, xmax, ymax))

            # 裁剪图像并调整比例
            imgs.append(cv2.resize(img[ymin:ymax, xmin:xmax], (64, 64)))
            labels.append(1)

        # 生成负例
        def hasOverlap(rect1, rect2):
            """判断两个矩形是否有交集"""
            x1, y1, x2, y2 = rect1
            a1, b1, a2, b2 = rect2
            return not (x2 <= a1 or x1 >= a2 or y2 <= b1 or y1 >= b2)

        num = num + 1
        while num:
            # 随机生成左上角点坐标
            x = random.randint(0, width - 64)
            y = random.randint(0, height - 64)
            newRect = (x, y, x + 64, y + 64)
            if all(not hasOverlap(newRect, targetRect) for targetRect in targetRects):
                imgs.append(img[y : y + 64, x : x + 64])
                labels.append(0)
                num -= 1
    return imgs, labels


def getHog(imgs):
    # 设置 HOG 参数
    win_size = (64, 64)  # 窗口大小
    block_size = (16, 16)  # 块大小(为单位进行归一化)
    block_stride = (8, 8)  # 块步长
    cell_size = (8, 8)  # 单元格大小
    nbins = 9  # 梯度方向的直方图 bins 数
    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins,
    )
    # 计算 HOG 特征
    hogFeatures = []
    for img in imgs:
        hogFeature = hog.compute(img)
        hogFeatures.append(hogFeature)
    return hogFeatures


path = "./train/"
imgs, labels = getTrainData(path)
hogFeatures = getHog(imgs)

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    hogFeatures, labels, test_size=0.2, random_state=233
)

# SVM 分类器
svm_classifier = SVC(kernel="rbf")
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

# 评估模型准确率
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy: {:.2f}%".format(accuracy * 100))
