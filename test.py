import os.path

from PIL import Image
import requests
import copy
import torch
import json

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle



# pretrained = "lmms-lab/llama3-llava-next-8b"
# model_name = "llava_llama3"

# pretrained = 'llava_logo_model1'
model_path = 'merge/baseline'
# model_base = 'liuhaotian/llava-v1.5-7b'

device = "cuda"
device_map = "cuda"

kwargs = {"device_map": device_map,
          'attn_implementation': 'flash_attention_2',
          'torch_dtype': torch.float16,}

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


import csv

# 保存csv
def save_csv(data, file_name):
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

conv_template = "v1" # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nYou are an intellectual property expert. Please check the image for potential copyright infringing elements, such as well-known brand logos or cartoon characters. If any are present, please provide the names of the brands or characters. If not, then reply that there are none."

with open('/root/autodl-tmp/logo_v1/test.json') as f:
    data = json.load(f)

results = []
for i in data:
    try:
        img_base = i['images'][0]
        print('*'*50)
        print(img_base)
        img_path = os.path.join('/root/autodl-tmp/', img_base)

        image = Image.open(img_path)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
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
        print('text_outputs', text_outputs)
        results.append([img_base, text_outputs[0]])
    except Exception as e:
        print(e)
save_csv(results, 'results_l.csv')
