import torch
from tqdm import tqdm
import random
from torchvision.utils import save_image

import matplotlib.pyplot as plt
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

    def __init__(self, args, model, targets, device='cuda:0', is_rtp=False):

        self.args = args
        self.model = model
        self.device = device
        self.is_rtp = is_rtp

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

    def attack_unconstrained(self, img, batch_size = 8, num_iter=2000, alpha=1/255):

        print('>>> batch_size:', batch_size)

        adv_noise = torch.rand_like(img).to(self.device) # [0,1]
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            x_adv = normalize(adv_noise).repeat(batch_size, 1, 1, 1)

            samples = {
                'image': x_adv,
                'text_input': [''] * batch_size,
                'text_output': batch_targets
            }

            target_loss = self.model(samples)['loss']
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

                with torch.no_grad():
                    print('>>> Sample Outputs')
                    print(self.model.generate({"image": x_adv, "prompt": ''},
                                     use_nucleus_sampling=True, top_p=0.9, temperature=1))

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def attack_constrained(self, img, batch_size = 8, num_iter=2000, alpha=1/255, epsilon = 128/255 ):

        print('>>> batch_size:', batch_size)

        adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data

        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()


        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            x_adv = x + adv_noise
            x_adv = normalize(x_adv).repeat(batch_size, 1, 1, 1)

            samples = {
                'image': x_adv,
                'text_input': [''] * batch_size,
                'text_output': batch_targets
            }

            target_loss = self.model(samples)['loss']
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

                with torch.no_grad():
                    print('>>> Sample Outputs')
                    print(self.model.generate({"image": x_adv, "prompt": ''},
                                              use_nucleus_sampling=True, top_p=0.9, temperature=1))

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