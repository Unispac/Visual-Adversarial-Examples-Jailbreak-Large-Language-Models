import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

from minigpt_utils import prompt_wrapper, generator

def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")


    parser.add_argument("--image_file", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args




# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

my_generator = generator.Generator(model=model)


# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
import json

datasets = []
with open('./harmful_corpus/red_teaming_prompts.jsonl', 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    datasets.append(result)



datasets = datasets[:1000] # shortlist 1k

print('size of shortlisted dataset : ', len(datasets))


prefix = prompt_wrapper.minigpt4_chatbot_prompt
img = Image.open(args.image_file).convert('RGB')
img_prompt = [vis_processor(img).unsqueeze(0).to(model.device)]



prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt])

out = []

from tqdm import tqdm

with torch.inference_mode():
    for i, data in tqdm(enumerate(datasets)):

        user_message = data['text']

        print(f" ----- {i} ----")
        print(" -- prompt: ---")
        print(prefix % user_message)

        prompt.update_text_prompt([prefix % user_message])
        response, _ = my_generator.generate(prompt)

        print(" -- response: ---")
        print(response)
        out.append({'prompt': user_message, 'response': response})
        print()


with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")