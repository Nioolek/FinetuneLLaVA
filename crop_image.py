import cv2
import numpy as np


def find_crop_coordinates(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # 使用Canny边缘检测
    # edges = cv2.Canny(gray, 50, 150)
    #
    # # 查找轮廓
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找图像的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，返回原始图像的大小
    if not contours:
        return 0, 0, image.shape[1], image.shape[0]

    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 获取边界框
    x, y, w, h = cv2.boundingRect(max_contour)

    return x, y, w, h


def crop_image(image_path, output_path):
    x, y, w, h = find_crop_coordinates(image_path)

    # 读取图像
    image = cv2.imread(image_path)
    print('org shape:', image.shape)
    print('new shape:', image[y:y + h, x:x + w].shape)

    # 裁剪图像
    cropped_image = image[y:y + h, x:x + w]

    # 保存裁剪后的图像
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to: {output_path}")


# 示例用法
image_path = r'I:\logo\logo0821\images\20th_century_fox\20th_century_fox1\yes\11.jpg'
output_path = 'cropped_image.jpg'
crop_image(image_path, output_path)