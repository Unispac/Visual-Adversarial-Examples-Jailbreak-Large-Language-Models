import torch
from llava_llama_2.conversation import conv_llava_llama_2
from llava_llama_2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_llama_2.mm_utils import tokenizer_image_token


def prepare_text_prompt(user_prompt):

    qs = DEFAULT_IMAGE_TOKEN + '\n' + user_prompt

    conv = conv_llava_llama_2.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt

# support batch implementation
class Prompt:
    # tokenization
    # turn to embeddings

    # padding? wait until targets have been appended
    # prepare labels? need to wait for targets

    def __init__(self, model, tokenizer, text_prompts=None, device='cuda:0'):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.text_prompts = text_prompts
        self.context_length = []
        self.input_ids = []
        self.do_tokenization(self.text_prompts)


    def do_tokenization(self, text_prompts):

        if text_prompts is None:
            self.input_ids = []
            self.context_length = []
            return

        input_ids = tokenizer_image_token(text_prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        self.input_ids = [input_ids]
        self.context_length = [input_ids.shape[1]]