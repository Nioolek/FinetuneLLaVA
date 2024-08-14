import sys
sys.path.append('.')
sys.path.append('llava')
import os
import json
import torch
import csv
from PIL import Image

# from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from logo_dict import sample_logo, sample_logo2, test_logo

model_path = 'merge/baseline2_train2'
result_csv_path = 'baseline2_train2.csv'

dataset_dir = '/root/autodl-tmp/data/logo0812'
img_dir = os.path.join(dataset_dir, 'images')
test_img_dir = '/root/autodl-tmp/data/logo0812_test_sample'
json_dir = os.path.join(dataset_dir, 'json')
train_json_path = os.path.join(json_dir, 'train1.json')
test_json_path = os.path.join(json_dir, 'all_test_images_sample.json')
test_none_json_path = os.path.join(json_dir, 'test_none.json')
test_amazon_brand_json_path = os.path.join(json_dir, 'test_amazon_brand.json')
device = "cuda"

train_logo = sample_logo
train_brand = [i.split('/')[0] for i in train_logo]



def init_model(model_path):
    device_map = "cuda"

    kwargs = {"device_map": device_map,
              'attn_implementation': 'flash_attention_2',
              'torch_dtype': torch.float16, }

    from transformers import AutoTokenizer, AutoConfig
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        config=cfg_pretrained,
        # device_map=device_map,
        **kwargs
    )

    vision_tower = model.get_vision_tower()
    print('vision_tower.is_loaded', vision_tower.is_loaded)
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return model, image_processor, context_len, tokenizer


def load_test_image_list():
    # data: {logo:[img1, img2, ...],}

    # 与训练集类别相同的数据
    train_logo_test_list = []
    test_logo_test_list = []
    amazon_test_list = []
    amazon_test_logo_list = []
    with open(test_json_path) as f:
        data = json.load(f)

    for logo, values in data.items():
        if logo in train_logo:
            logo_in_train = True
        else:
            logo_in_train = False
        brand = logo.split('/')[0]
        if brand in train_brand:
            brand_in_train = True
        else:
            brand_in_train = False
        if logo_in_train or brand_in_train:
            for img in values:
                # logo, img, logo是否在训练集，brand是否在训练集，是否是amazon验证集
                img_path = os.path.join(test_img_dir, logo, 'yes', img)
                train_logo_test_list.append([logo, img_path, logo_in_train, brand_in_train, False])
        # 测试集数据
        if logo in test_logo:
            for img in values:
                img_path = os.path.join(test_img_dir, logo, 'yes', img)
                test_logo_test_list.append([logo, img_path, False, False, False])

    # 亚马逊无品牌数据
    with open(test_none_json_path) as f:
        data = json.load(f)
    for logo, values in data.items():
        for img in values:
            img_path = os.path.join(img_dir, logo, 'yes', img)
            amazon_test_list.append([logo, img_path, False, False, True])

    # 亚马逊有品牌数据
    with open(test_amazon_brand_json_path) as f:
        data = json.load(f)
    for logo, values in data.items():
        for img in values:
            img_path = os.path.join(img_dir, logo, 'yes', img)
            amazon_test_logo_list.append([logo, img_path, False, False, True])

    return train_logo_test_list + test_logo_test_list + amazon_test_list + amazon_test_logo_list
    # return test_logo_test_list + amazon_test_list + amazon_test_logo_list


class WriteCsv():
    def __init__(self, save_path, header=['logo', 'img_path','logo in train', 'brand in train', 'is amazom', 'result']):
        self.save_path = save_path
        self.csv_file = open(self.save_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.write_row(header)

    def write_row(self, row):
        self.writer.writerow(row)

    def close(self):
        self.csv_file.close()


def get_result(model, logo, imgbase, image_processor, question, tokenizer, msg):
    img_path = os.path.join(img_dir, logo, 'yes', imgbase)
    image = Image.open(img_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    writer.write_row([logo, os.path.basename(imgbase), *msg, text_outputs[0]])
    print('logo: ', logo, 'imgbase: ', os.path.basename(imgbase), 'output: ', text_outputs[0])



if __name__ == '__main__':

    conv_template = "v1" # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\nYou are an intellectual property expert. Please check the image for potential copyright infringing elements, such as well-known brand logos or cartoon characters. If any are present, please provide the names of the brands or characters. If not, then reply that there are none."

    writer = WriteCsv(result_csv_path)
    model, image_processor, context_len, tokenizer = init_model(model_path)
    data = load_test_image_list()

    # 推理
    for logo, imgbase, *msg in data:
        get_result(model, logo, imgbase, image_processor, question, tokenizer, msg)
