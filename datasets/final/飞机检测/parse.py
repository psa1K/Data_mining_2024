import os
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog as skimage_hog

random.seed(233)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

WINDOW_SIZE = (64, 64)  # 窗口大小


def get_train_data(path):
    """读取训练数据，返回正样本（飞机）与负样本（非飞机）的裁剪图及其标签。"""
    imgs = []
    labels = []
    for xml_name in os.listdir(path + "Annotation/xml/"):
        tree = ET.parse(path + "Annotation/xml/" + xml_name)
        root = tree.getroot()
        # 获取图像尺寸
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        # 获取图像名
        img_name = root.find("filename").text
        # 读取图像
        img = cv2.imread(path + "JPEGImages/" + img_name, cv2.IMREAD_GRAYSCALE)

        # 获取所有边界框
        target_rects = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = max(0, int(bndbox.find("xmin").text))
            ymin = max(0, int(bndbox.find("ymin").text))
            xmax = max(0, int(bndbox.find("xmax").text))
            ymax = max(0, int(bndbox.find("ymax").text))
            target_rects.append((xmin, ymin, xmax, ymax))

            # 裁剪图像并调整比例（正样本）
            imgs.append(cv2.resize(img[ymin:ymax, xmin:xmax], WINDOW_SIZE))
            labels.append(1)

        # 生成负样本：每张图取 len(target_rects)+1 个不与任何边界框重叠的窗口
        def has_overlap(rect1, rect2):
            """判断两个矩形是否有交集"""
            x1, y1, x2, y2 = rect1
            a1, b1, a2, b2 = rect2
            return not (x2 <= a1 or x1 >= a2 or y2 <= b1 or y1 >= b2)

        negative_count = len(target_rects) + 1
        while negative_count:
            # 随机生成左上角点坐标
            x = random.randint(0, max(0, width - WINDOW_SIZE[0]))
            y = random.randint(0, max(0, height - WINDOW_SIZE[1]))
            new_rect = (x, y, x + WINDOW_SIZE[0], y + WINDOW_SIZE[1])
            if all(
                not has_overlap(new_rect, target_rect) for target_rect in target_rects
            ):
                imgs.append(img[y : y + WINDOW_SIZE[1], x : x + WINDOW_SIZE[0]])
                labels.append(0)
                negative_count -= 1
    return imgs, labels


def get_hog(imgs):
    """计算图像的 HOG 特征。

    参数：cell=8x8、block=2x2 cells、9 方向直方图、L2-Hys 归一化。
    """
    return [
        skimage_hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        for img in imgs
    ]


def sliding_window(image, step_size, window_size):
    """滑动窗口生成器，遍历图像上的所有窗口。"""
    for y in range(0, image.shape[0] - window_size[0], step_size):
        for x in range(0, image.shape[1] - window_size[1], step_size):
            yield (x, y, image[y : y + window_size[1], x : x + window_size[0]])


def detect_aircraft(image_path, svm_classifier, step_size=16):
    """在新图像上滑动窗口检测飞机，返回命中窗口的左上角坐标列表。"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pos_points = []
    for x, y, window in sliding_window(img, step_size, WINDOW_SIZE):
        hog_feature = get_hog([window])
        if svm_classifier.predict(hog_feature)[0] == 1:
            pos_points.append((x, y))
    return img, pos_points


def main():
    train_data_path = "./train/"
    imgs, labels = get_train_data(train_data_path)
    hog_features = get_hog(imgs)

    # 数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        hog_features, labels, test_size=0.2, random_state=233
    )

    # SVM 分类器
    svm_classifier = SVC(kernel="rbf")
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    # 评估模型准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("SVM Accuracy: {:.2f}%".format(accuracy * 100))

    # 在示例图像上检测飞机
    img, pos_points = detect_aircraft(
        train_data_path + "JPEGImages/aircraft_4.jpg", svm_classifier
    )
    for x, y in pos_points:
        cv2.rectangle(
            img,
            (x, y),
            (x + WINDOW_SIZE[0], y + WINDOW_SIZE[1]),
            (0, 255, 0),
            2,
        )
    cv2.imwrite("detected_aircraft_4.jpg", img)
    print("Detected {} aircraft, result saved to detected_aircraft_4.jpg".format(
        len(pos_points)
    ))


if __name__ == "__main__":
    main()
