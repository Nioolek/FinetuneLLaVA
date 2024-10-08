from PIL import Image
import requests
import copy
import torch

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

# img_path = '../../logo_v1/ALMAY1/yes/IMG_1224.PNG'
# img_path = '/root/autodl-tmp/data/logo0812/images/Dior/Dior1/yes/IMG_3761.PNG'
img_path = '/root/autodl-tmp/data/logo0812/images/Mickey_Mouse/Mickey_Mouse1/yes/251.jpg'
image = Image.open(img_path)

# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, model_base, model_name, device_map=device_map, use_flash_attn=True) # Add any other thing you want to pass in llava_model_args
#
# model.eval()
# model.tie_weights()
#
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "v1" # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nYou are an intellectual property expert. Please check the image for potential copyright infringing elements, such as well-known brand logos or cartoon characters. If any are present, please provide the names of the brands or characters. If not, then reply that there are none."
conv = conv_templates[conv_template].copy()
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

print('xxx', input_ids, image_tensor, image_sizes)
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=256,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)
# The image shows a radar chart, also known as a spider chart or a web chart, which is a type of graph used to display multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. Each axis represents a different variable, and the values are plotted along each axis and connected to form a polygon.\n\nIn this particular radar chart, there are several axes labeled with different variables, such as "MM-Vet," "LLaVA-Bench," "SEED-Bench," "MMBench-CN," "MMBench," "TextVQA," "VizWiz," "GQA," "BLIP-2," "InstructBLIP," "Owen-VL-Chat," and "LLaVA-1.5." These labels suggest that the chart is comparing the performance of different models or systems across various benchmarks or tasks, such as machine translation, visual question answering, and text-based question answering.\n\nThe chart is color-coded, with each color representing a different model or system. The points on the chart are connected to form a polygon, which shows the relative performance of each model across the different benchmarks. The closer the point is to the outer edge of the