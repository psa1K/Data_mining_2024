import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 中文
plt.rcParams["axes.unicode_minus"] = False  # 正负号

# 更改当前工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
c_path = "./cloud/"
f_path = "./forest/"

os.makedirs("./output/glcm/", exist_ok=True)  # 创建输出文件夹
os.makedirs("./output/lbp/", exist_ok=True)

c_name = [c_path + i for i in os.listdir(c_path)]
f_name = [f_path + i for i in os.listdir(f_path)]


def show_grey(root=c_name):
    for n, i in enumerate(c_name):
        plt.subplot(2, 5, n + 1)
        img = Image.open(i)
        img = img.convert("L")  # 转为灰度图
        plt.imshow(img, cmap="gray")  # 显示灰度图
        plt.axis("off")  # 不显示坐标轴
    plt.show()


def get_mat(root):
    for n, i in enumerate(root):
        # plt.subplot(2, 5, n + 1)
        img = Image.open(i)
        img = img.convert("L")  # 转为灰度图
        img = np.array(img)  # 转为ndarray
        yield i, img


# 计算灰度共生矩阵(gray-level co-occurrence matrix)
def get_glcm(img, angle=0, distance=1, gray_levels=256):
    # 归一化到0-gray_levels灰度级
    img = (img / (np.max(img) + 1e-5) * (gray_levels - 1)).astype(int)
    h, w = img.shape  # 获取图像的尺寸
    glcm = np.zeros((gray_levels, gray_levels), dtype=np.int64)

    for i in range(h):
        for j in range(w):
            match angle:
                case 0:
                    if j + distance < w:
                        glcm[img[i, j], img[i, j + distance]] += 1
                case 90:
                    if i + distance < h:
                        glcm[img[i, j], img[i + distance, j]] += 1
                case 45:
                    if i + distance < h and j + distance < w:
                        glcm[img[i, j], img[i + distance, j + distance]] += 1
                case 135:
                    if i + distance < h and j - distance >= 0:
                        glcm[img[i, j], img[i + distance, j - distance]] += 1

    glcm = glcm / np.sum(glcm)  # 归一化
    return glcm


# 绘制灰度共生矩阵
def get_glcm_fig(root, angles=[0, 45, 90, 135]):
    for i, img in get_mat(root):
        glcms = []  # 存储各角度的 GLCM 图像
        for angle in angles:
            glcm = get_glcm(img, angle=angle)
            glcm = (glcm / glcm.max() * 255).astype(np.uint8)
            glcms.append((Image.fromarray(glcm), f"Angle: {angle}°"))  # 保存图像和标题

        # 计算单张 GLCM 图像的大小
        width, height = glcms[0][0].size
        title_height = 20  # 为标题预留的高度

        # 2x2 布局，标题增加额外空间
        canvas = Image.new("L", (2 * width, 2 * (height + title_height)), "white")

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for idx, (glcm, title) in enumerate(glcms):
            x_offset = (idx % 2) * width
            y_offset = (idx // 2) * (height + title_height)

            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x_offset + (width - text_width) // 2
            draw.text((text_x, y_offset), title, fill="black", font=font)

            canvas.paste(glcm, (x_offset, y_offset + title_height))

        canvas.save(f"./output/glcm/{i}")


# 计算局部二值模式(local binary pattern)
def get_lbp(img):
    h, w = img.shape  # 获取图像的尺寸
    lbp = np.zeros((h - 2, w - 2), dtype=np.int64)  # 忽略边缘像素
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = img[i, j]
            neighbors = [
                img[i, j - 1],
                img[i + 1, j - 1],
                img[i + 1, j],
                img[i + 1, j + 1],
                img[i, j + 1],
                img[i - 1, j + 1],
                img[i - 1, j],
                img[i - 1, j - 1],
            ]
            code = 0
            for idx, neighbor in enumerate(neighbors):
                if neighbor > center:
                    code += 2**idx
            lbp[i - 1, j - 1] = code
    return lbp


# 绘制局部二值模式
def get_lbp_fig(root):
    for i, img in get_mat(root):
        lbp = get_lbp(img)
        Image.fromarray(lbp.astype(np.uint8)).save(f"./output/lbp/{i}")


def main():
    roots = [c_name, f_name]
    for root in roots:
        get_glcm_fig(root)
        get_lbp_fig(root)


if __name__ == "__main__":
    main()
