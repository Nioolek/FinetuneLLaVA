import base64
import os
import requests
import json
import csv
import openai
import random
from tqdm import tqdm

import xml.etree.ElementTree as ET

from logo_dict import logo2brand


def parse_voc_annotation(xml_filepath):
    # 解析XML文件
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    # 提取基本信息
    folder = root.find('folder').text
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 初始化一个列表来存储所有目标信息
    objects = []

    # 遍历所有目标
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 提取目标的其他信息
        name = obj.find('name').text
        pose = obj.find('pose').text if obj.find('pose') is not None else 'Unspecified'
        truncated = int(obj.find('truncated').text)
        difficult = int(obj.find('difficult').text)

        # 将目标信息存储为字典
        object_info = {
            'name': name,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
            'bndbox': {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }
        }

        objects.append(object_info)

        # 返回提取的信息
    return {
        'folder': folder,
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


openai.api_key = '???'
openai.api_base = 'https://gf.gpt.ge/v1'


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# IMAGE_PATH = r'H:\temp\ALMAY\ALMAY1\yes\IMG_1276_crop.PNG'



have_img_list = []
with open('req_gpt4o.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        have_img_list.append(row[0])


writer = csv.writer(open('req_gpt4o1.csv', 'w', newline=''))

img_list = []
for a, b, c_list in os.walk(r'H:\temp'):
    for c in c_list:
        if c.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(a, c)
            xml_path = os.path.splitext(img_path)[0] + '.xml'

            img_list.append([img_path, xml_path])


replace_dict = {}

random.shuffle(img_list)
for IMAGE_PATH, XML_PATH in tqdm(img_list, total=len(img_list), desc='Processing'):
    if IMAGE_PATH in have_img_list:
        # print('have', IMAGE_PATH)
        continue

    info = parse_voc_annotation(XML_PATH)
    objects = info['objects']
    names = set([obj['name'] for obj in objects])
    new_names = []
    for name in names:
        if name in logo2brand:
            new_names.append(logo2brand[name])
        elif name in replace_dict:
            new_names.append(replace_dict[name])
        elif name.startswith('my/'):
            new_names.append(name.split('/')[-1])
        else:
            print(name)
            raise NotImplementedError
    new_names = sorted(list(set(new_names)))

    # msg = 'This image includes some well-known brands or cartoon characters, including %s. Please briefly describe the main features of the product in this image.' % ', '.join(new_names)
    try:
        base64_image = encode_image(IMAGE_PATH)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Help me with my task!"},
                {"role": "user", "content": [
                    # {"type": "text", "text": "Describe this image in detail."},
                    {"type": "text", "text": 'This image includes some well-known brands or cartoon characters, including %s. Please briefly describe the main features of the product in this image.' % ', '.join(new_names)},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpg;base64,{base64_image}"}
                     }
                ]},
            ],
            max_tokens=2560,
        )
        # print(response.choices[0].message.content)
        content = response.choices[0].message.content
        writer.writerow([IMAGE_PATH, content])
    except Exception as e:
        continue












# org img detail
# The image shows a bottle of Almay Skin Perfecting™ Comfort Matte Foundation placed against a textured gray background. The foundation is labeled with several key pieces of information:
#
# - The brand name "ALMAY" is prominently displayed at the top of the bottle.
# - The product name is "Skin Perfecting™ Comfort Matte Foundation."
# - The shade of the foundation is "220 Warm Cashew (Cajou Chaud)."
# - The product claims to be hypoallergenic and fragrance-free.
# - Additional product information includes "ALL DAY WEAR", "Oil Free," and "All Skin Types" written on the black cap.
# - At the bottom, the bottle indicates that the quantity is 1.0 fl oz / 30 ml.
#
# The bottle itself is made of clear plastic, allowing the color of the foundation to be visible. The text on the bottle is black and the shade name is on a beige background, matching the color of the foundation.

# org img brief
# The product in the image is Almay Skin Perfecting Comfort Matte Foundation with the following key features:
#
# - Shade: 220 Warm Cashew
# - Hypoallergenic
# - Fragrance-free
# - Oil-free
# - Suitable for all skin types
# - All-day wear
#
# The foundation is intended to provide a matte finish while being gentle on sensitive skin. The bottle contains 30 ml (1.0 fl oz) of product.


# crop img detail
# The image shows a bottle of Almay Skin Perfecting Comfort Matte Foundation placed on a textured gray surface. The bottle is made of clear glass, allowing the light tan foundation inside to be visible. The bottle has a black cap, and the text on it states:
#
# "ALL DAY WEAR
# Longue Tenue
# Oil Free
# Sans Huile
# All Skin Types
# Tous Types de peaux"
#
# The label on the front of the bottle includes the following text:
#
# "ALMAY
# Skin Perfecting™
# Comfort Matte Foundation
# Fond de teint mat confort
# 220
# WARM CASHEW
# Cajou Chaud
#
# HYPOALLERGENIC
# Hypoallergénique
# FRAGRANCE FREE
# Sans parfum
#
# 1.0 fl oz / 30 mL"
#
# There is also a triangular design in the center with metallic and reflective sections, and the foundation appears to be shade number 220, named "Warm Cashew."


# crop img brief
# The product in the image is Almay Skin Perfecting Comfort Matte Foundation. Its main features include:
#
# 1. **Long Wear**: Provides all-day wear.
# 2. **Oil-Free**: Suitable for those who prefer or need oil-free products.
# 3. **For All Skin Types**: Formulated to be suitable for all skin types.
# 4. **Shade**: The specific shade indicated is 220 Warm Cashew.
# 5. **Hypoallergenic**: Designed to minimize the risk of allergic reactions.
# 6. **Fragrance-Free**: Does not contain fragrances, catering to those with sensitive skin or who prefer unscented products.
# 7. **Size**: The bottle contains 1.0 fluid ounces (30 mL) of product.