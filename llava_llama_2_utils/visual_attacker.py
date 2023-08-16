import torch
from tqdm import tqdm
import random
from llava_llama_2_utils import prompt_wrapper, generator
from torchvision.utils import save_image


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns




def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class Attacker:

    def __init__(self, args, model, tokenizer, targets, device='cuda:0', is_rtp=False, image_processor=None):

        self.args = args
        self.model = model
        self.tokenizer= tokenizer
        self.device = device
        self.is_rtp = is_rtp

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        self.image_processor = image_processor

    def attack_unconstrained(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255):

        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)

        adv_noise = torch.rand_like(img).cuda() # [0,1]
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        prompt = prompt_wrapper.Prompt( self.model, self.tokenizer, text_prompts=text_prompt, device=self.device )

        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            x_adv = normalize(adv_noise)

            target_loss = self.attack_loss(prompt, x_adv, batch_targets)

            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = normalize(adv_noise)
                response = my_generator.generate(prompt, x_adv)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def attack_constrained(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255, epsilon = 128/255 ):

        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)


        adv_noise = torch.rand_like(img).cuda() * 2 * epsilon - epsilon

        x = denormalize(img).clone().cuda()
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise = adv_noise.cuda()
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)

        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            x_adv = x + adv_noise
            x_adv = normalize(x_adv)

            target_loss = self.attack_loss(prompt, x_adv, batch_targets)
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)
                response = my_generator.generate(prompt, x_adv)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def plot_loss(self):

        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

        torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

    def attack_loss(self, prompts, images, targets):

        context_length = prompts.context_length
        context_input_ids = prompts.input_ids
        batch_size = len(targets)

        if len(context_input_ids) == 1:
            context_length = context_length * batch_size
            context_input_ids = context_input_ids * batch_size

        images = images.repeat(batch_size, 1, 1, 1)

        assert len(context_input_ids) == len(targets), f"Unmathced batch size of prompts and targets {len(context_input_ids)} != {len(targets)}"


        to_regress_tokens = [ torch.as_tensor([item[1:]]).cuda() for item in self.tokenizer(targets).input_ids] # get rid of the default <bos> in targets tokenization.


        seq_tokens_length = []
        labels = []
        input_ids = []

        for i, item in enumerate(to_regress_tokens):

            L = item.shape[1] + context_length[i]
            seq_tokens_length.append(L)

            context_mask = torch.full([1, context_length[i]], -100,
                                      dtype=to_regress_tokens[0].dtype,
                                      device=to_regress_tokens[0].device)
            labels.append( torch.cat( [context_mask, item], dim=1 ) )
            input_ids.append( torch.cat( [context_input_ids[i], item], dim=1 ) )

        # padding token
        pad = torch.full([1, 1], 0,
                         dtype=to_regress_tokens[0].dtype,
                         device=to_regress_tokens[0].device).cuda() # it does not matter ... Anyway will be masked out from attention...


        max_length = max(seq_tokens_length)
        attention_mask = []

        for i in range(batch_size):

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]

            padding_mask = (
                torch.full([1, num_to_pad], -100,
                       dtype=torch.long,
                       device=self.device)
            )
            labels[i] = torch.cat( [labels[i], padding_mask], dim=1 )

            input_ids[i] = torch.cat( [input_ids[i],
                                       pad.repeat(1, num_to_pad)], dim=1 )
            attention_mask.append( torch.LongTensor( [ [1]* (seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        labels = torch.cat( labels, dim=0 ).cuda()
        input_ids = torch.cat( input_ids, dim=0 ).cuda()
        attention_mask = torch.cat(attention_mask, dim=0).cuda()

        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels,
                images=images.half(),
            )
        loss = outputs.loss

        return loss