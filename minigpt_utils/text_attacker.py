import torch
from tqdm import tqdm
import random
from minigpt_utils import prompt_wrapper, generator
from torchvision.utils import save_image
import numpy as np
from copy import deepcopy
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns


class Attacker:

    def __init__(self, args, model, targets, device='cuda:0'):

        self.args = args
        self.model = model
        self.device = device

        self.targets = targets # targets that we want to promte likelihood
        self.loss_buffer = []
        self.num_targets = len(self.targets)

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.llama_tokenizer.padding_side = "right"

    def get_vocabulary(self):

        vocab_dicts = self.model.llama_tokenizer.get_vocab()
        vocabs = vocab_dicts.keys()

        single_token_vocabs = []
        single_token_vocabs_embedding = []
        single_token_id_to_vocab = dict()
        single_token_vocab_to_id = dict()

        cnt = 0

        for item in vocabs:
            tokens = self.model.llama_tokenizer(item, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if tokens.shape[1] == 1:

                single_token_vocabs.append(item)
                emb = self.model.llama_model.model.embed_tokens(tokens)
                single_token_vocabs_embedding.append(emb)

                single_token_id_to_vocab[cnt] = item
                single_token_vocab_to_id[item] = cnt

                cnt+=1

        single_token_vocabs_embedding = torch.cat(single_token_vocabs_embedding, dim=1).squeeze()

        self.vocabs = single_token_vocabs
        self.embedding_matrix = single_token_vocabs_embedding.to(self.device)
        self.id_to_vocab = single_token_id_to_vocab
        self.vocab_to_id = single_token_vocab_to_id


    def hotflip_attack(self, grad, token,
                       increase_loss=False, num_candidates=1):

        token_id = self.vocab_to_id[token]
        token_emb = self.embedding_matrix[token_id] # embedding of current token

        scores = ((self.embedding_matrix - token_emb) @ grad.T).squeeze(1)

        if not increase_loss:
            scores *= -1  # lower versus increase the class probability.

        _, best_k_ids = torch.topk(scores, num_candidates)
        return best_k_ids.detach().cpu().numpy()

    def wrap_prompt(self, text_prompt_template, adv_prompt, queries, batch_size):

        text_prompts = text_prompt_template % (adv_prompt + ' | ' + queries)

        prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=[text_prompts], img_prompts=[[]])

        prompt.context_embs[0] = prompt.context_embs[0].detach().requires_grad_(True)
        prompt.context_embs = prompt.context_embs * batch_size

        return prompt

    def wrap_prompt_simple(self, text_prompt_template, adv_prompt, batch_size):

        text_prompts = text_prompt_template % (adv_prompt) # insert the adversarial prompt

        prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=[text_prompts], img_prompts=[[]])

        prompt.context_embs[0] = prompt.context_embs[0].detach().requires_grad_(True)
        prompt.context_embs = prompt.context_embs * batch_size

        return prompt

    def update_adv_prompt(self, adv_prompt_tokens, idx, new_token):
        next_adv_prompt_tokens = deepcopy(adv_prompt_tokens)
        next_adv_prompt_tokens[idx] = new_token
        next_adv_prompt = ' '.join(next_adv_prompt_tokens)
        return next_adv_prompt_tokens, next_adv_prompt



    def attack(self, text_prompt_template, offset, batch_size = 8, num_iter=2000):

        print('>>> batch_size: ', batch_size)

        my_generator = generator.Generator(model=self.model)

        self.get_vocabulary()
        vocabs, embedding_matrix = self.vocabs, self.embedding_matrix

        trigger_token_length = 32 # equivalent to
        adv_prompt_tokens = random.sample(vocabs, trigger_token_length)
        adv_prompt = ' '.join(adv_prompt_tokens)

        st = time.time()

        for t in tqdm(range(num_iter+1)):

            for token_to_flip in range(0, trigger_token_length): # for each token in the trigger

                batch_targets = random.sample(self.targets, batch_size)

                prompt = self.wrap_prompt_simple(text_prompt_template, adv_prompt, batch_size)

                target_loss = self.attack_loss(prompt, batch_targets)
                loss = target_loss # to minimize
                loss.backward()

                print('[adv_prompt]', adv_prompt)
                print("target_loss: %f" % (target_loss.item()))
                self.loss_buffer.append(target_loss.item())

                tokens_grad = prompt.context_embs[0].grad[:, token_to_flip+offset, :]
                candidates = self.hotflip_attack(tokens_grad, adv_prompt_tokens[token_to_flip],
                                            increase_loss=False, num_candidates=self.args.n_candidates)

                self.model.zero_grad()

                # try all the candidates and pick the best
                # comparing candidates does not require gradient computation
                with torch.no_grad():
                    curr_best_loss = 999999
                    curr_best_trigger_tokens = None
                    curr_best_trigger = None

                    for cand in candidates:
                        next_adv_prompt_tokens, next_adv_prompt = self.update_adv_prompt(adv_prompt_tokens,
                                                                    token_to_flip, self.id_to_vocab[cand])
                        prompt = self.wrap_prompt_simple(text_prompt_template, next_adv_prompt, batch_size)

                        next_target_loss = self.attack_loss(prompt, batch_targets)
                        curr_loss = next_target_loss  # to minimize

                        if curr_loss < curr_best_loss:
                            curr_best_loss = curr_loss
                            curr_best_trigger_tokens = next_adv_prompt_tokens
                            curr_best_trigger = next_adv_prompt

                    # Update overall best if the best current candidate is better
                    if curr_best_loss < loss:
                        adv_prompt_tokens = curr_best_trigger_tokens
                        adv_prompt = curr_best_trigger
                print('(update: %f minutes)' % ((time.time() - st) / 60))

            self.plot_loss()

            if True:
                print('######### Output - Iter = %d ##########' % t)
                prompt = self.wrap_prompt_simple(text_prompt_template, adv_prompt, batch_size)
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt)

                print('[prompt]', prompt.text_prompts[0])
                print('>>>', response)
                
        return adv_prompt

    def plot_loss(self):

        sns.set_theme()

        num_iters = len(self.loss_buffer)

        num_iters = min(num_iters, 5000)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer[:num_iters], label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

        torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))


    def attack_loss(self, prompts, targets):

        context_embs = prompts.context_embs
        assert len(context_embs) == len(targets), "Unmathced batch size of prompts and targets, the length of context_embs is %d, the length of targets is %d" % (len(context_embs), len(targets))

        batch_size = len(targets)

        self.model.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.model.llama_tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False
        ).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        bos = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.bos_token_id
        bos_embs = self.model.llama_model.model.embed_tokens(bos)

        pad = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.pad_token_id
        pad_embs = self.model.llama_model.model.embed_tokens(pad)


        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )


        pos_padding = torch.argmin(T, dim=1) # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):

            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i:i+1, :target_length])
            input_embs.append(to_regress_embs[i:i+1, :target_length]) # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):

            # masked out the context from loss computation
            context_mask =(
                torch.ones([1, context_tokens_length[i] + 1],
                       dtype=torch.long).to(self.device).fill_(-100)  # plus one for bos
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = (
                torch.ones([1, num_to_pad],
                       dtype=torch.long).to(self.device).fill_(-100)
            )

            targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
            input_embs[i] = torch.cat( [bos_embs, context_embs[i], input_embs[i],
                                        pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
            attention_mask.append( torch.LongTensor( [[1]* (1+seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        targets = torch.cat( targets_mask, dim=0 ).to(self.device)
        inputs_embs = torch.cat( input_embs, dim=0 ).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)


        outputs = self.model.llama_model(
                inputs_embeds=inputs_embs,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return loss
