<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Visual Adversarial Examples Jailbreak<br>Aligned Large Language Models </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://unispac.github.io/" target="_blank" style="text-decoration: none;">Xiangyu Qi<sup>1,*</sup></a>&nbsp;,&nbsp;
    <a href="https://hackyhuang.github.io/" target="_blank" style="text-decoration: none;">Kaixuan Huang<sup>1,*</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=rFC3l6YAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Ashwinee Panda<sup>1</sup></a><br>
  <a href="https://www.peterhenderson.co/" target="_blank" style="text-decoration: none;">Peter Henderson<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://mwang.princeton.edu/" target="_blank" style="text-decoration: none;">Mengdi Wang<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://www.princeton.edu/~pmittal/" target="_blank" style="text-decoration: none;">Prateek Mittal<sup>1</sup></a>&nbsp;&nbsp; 
    <br/> 
<sup>*</sup>Equal Contribution<br>
Princeton University<sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;Stanford University<sup>2</sup><br/> 
</p>




<p align='center';>
<b>
<em>AAAI (Oral), 2024</em> <br>
</b>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2306.13213" target="_blank" style="text-decoration: none;">arXiv</a>&nbsp;
</b>
</p>


------------



***!!! Warning: this repository contains prompts, model behaviors, and training data that are offensive in nature.***

<br>

<br>

![](assets/attack.png)





------------

**Overview: A single visual adversarial example can jailbreak MiniGPT-4.** 

> Note
>
> 1. For each instruction below, we've sampled 100 random outputs, calculating the refusal and obedience ratios via manual inspection. A representative, redacted output is showcased for each.
> 2.  We use ɛ = 16/255 in the following demo.

![](assets/human_race.png)

![](assets/gender.png)

![](assets/race.png)

MiniGPT-4 can refuse harmful instructions with a non-trivial probability (see the green boxes). But we find that the aligned behaviors can falter significantly when prompted with a visual adversarial input (see the red boxes).

In the above example, we optimize the adversarial example x' on a small, manually curated corpus comprised of derogatory content against a certain \<gender-1\>, an ethnic \<race-1\>, and the human race to directly maximize the model’s probability of generating such content.

Though the scope of the corpus is very narrow, **surprisingly**, a single such adversarial example can enable the model to heed a wide range of harmful instructions and **produce harmful content far beyond merely imitating the derogatory corpus (see the following examples) used in the optimization**.

![](assets/religious-1.png)

![](assets/religious-2.png)

![](assets/crime.png)

**Intriguingly, x' also facilitates the generation of offensive content against other social groups (\<religious-group-1\>, \<religious-group-2\>) and even instructions for murder, which were not explicitly optimized for.** 

> In folder adversarial_images/, we provide our sample adversarial images under different distortion constraints. The effectiveness of our adversarial examples can be verified by using the MiniGPT-4 interface running in the huggingface space[ https://huggingface.co/spaces/Vision-CAIR/minigpt4](https://huggingface.co/spaces/Vision-CAIR/minigpt4).

----------

<br>

<br>

# Step-by-Step Instructions for Reimplementing Our Experiments on MiniGPT-4
> Note: a single A100 80G GPU is sufficient to launch the following experiments.
<br>

### Installation

We take MiniGPT-4 (13B) as the sandbox to showcase our attacks. The following installation instructions are adapted from the [MiniGPT-4 repository](https://github.com/Vision-CAIR/MiniGPT-4).

**1. Set up the environment**

```bash
git clone https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models.git

cd Visual-Adversarial-Examples-Jailbreak-Large-Language-Models

conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare the pretrained weights for MiniGPT-4**

> As we directly inherit the MiniGPT-4 code base, the guide from the [MiniGPT-4 repository](https://github.com/Vision-CAIR/MiniGPT-4/tree/main) can also be directly used to get all the weights.

* **Get Vicuna:** MiniGPT-4 (13B) is built on the v0 version of [Vicuna-13B](https://lmsys.org/blog/2023-03-30-vicuna/). Please refer to this [guide](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/PrepareVicuna.md) from the MiniGPT-4 repository to get the weights of Vicuna.

  Then, set the path to the vicuna weight in the model config file [here](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/minigpt4/configs/models/minigpt4.yaml#L16) at Line 16.

* **Get MiniGPT-4 (the 13B version) checkpoint**: download from [here](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link). 

  Then, set the path to the pretrained checkpoint in the evaluation config file in [eval_configs/minigpt4_eval.yaml](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/eval_configs/minigpt4_eval.yaml#L11) at Line 11.

<br>

### Generate Visual Adversarial Examples


Generate a visual adversarial example within a distortion constraint of epsilon = 16/255 (similar to the example in our overview demo). The final adversarial examples will be saved to `$save_dir/bad_prompt.bmp`, and we also save intermediate checkpoints every 100 iterations.

The argument can be adjusted (e.g., --eps=32,64,128) to evaluate the effectiveness of the attacks under different distortion budgets.

```bash
python minigpt_visual_attack.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --n_iters 5000 --constrained --eps 16 --alpha 1 --save_dir visual_constrained_eps_16
```

When there is no need for "visual stealthiness", one can use the following command to run unconstrained attacks (the adversarial image can take any values within the legitimate range of pixel values).

```bash
python minigpt_visual_attack.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --n_iters 5000  --alpha 1 --save_dir visual_unconstrained
```

<br>

### Evaluation

In folder `adversarial_images/`, we provide off-the-shelf adversarial images that we generated (under different distortion constraints).

To verify the effectiveness of our adversarial examples:

* Play with the web-based interface of MiniGPT-4:

  * Huggingface Space: https://huggingface.co/spaces/Vision-CAIR/minigpt4

  * Launch a local demo: 

    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

* Testing on a diverse set of 40 manually curated harmful instructions

  (Warning: this will involve materials that are offensive in nature)

  ```bash
  python minigpt_test_manual_prompts_visual_llm.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_path  adversarial_images/prompt_unconstrained.bmp
  ```

  The argument `--image_path` can be customized to the path of any input image.

* Testing on the RealToxicityPrompts Dataset

    * Download the [RealToxicityPrompts dataset](https://allenai.org/data/real-toxicity-prompts) and copy/soft-link the dataset file to `./rtp_prompts.jsonl`. 

      The `inference.py` will read the dataset, filter out prompts with `challenging = True` and ask the model to generate the continuation for each prompt. 

      ```bash
      python minigpt_inference.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file  adversarial_images/prompt_unconstrained.bmp --output_file result.jsonl
      ```

      The argument `--image_file` can be customized to the path of any input image, `--output_file` indicates where to save the generations.

    * Request and place your [Perspective API](https://perspectiveapi.com/) key in `.perspective_api_key`.

      The `get_metric.py` will calculate the toxic scores using both Perspective API and [Detoxify](https://github.com/unitaryai/detoxify).

      ```bash
      python get_metric.py --input result.jsonl --output result_eval.jsonl
      ```

<br>

### Generate Textual Adversarial Examples

We also provide codes for optimizing adversarial text tokens w.r.t. the same attack targets as our visual attacks. A running example: 

```bash
python minigpt_textual_attack.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --n_iters 5000 --n_candidates 50 --save_dir textual_unconstrained
```

<br><br>

# Attacks on Other Models

We also implement our attacks on two other open-sourced VLMs, including [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) and [LLaVA](https://github.com/haotian-liu/LLaVA). To launch experiments on these models, we suggest users to create a separate conda environment for each model and install dependencies following the instructions of the original repositories of these models.

### Instruct BLIP

* Set up the environment and prepare model checkpoints following the instructions of the original repository at: https://github.com/salesforce/LAVIS/tree/main/projects/instructblip

  * Choose the 13B version model built upon vicuna13b, i.e., blip2_vicuna_instruct-vicuna13b
  * Download the version 1.1 model of vicuna-13b to the directory `./ckpts/vicuna-13b-v1.1`

* Attack

  ```bash
  python -u instructblip_visual_attack.py --n_iters 5000 --constrained --save_dir results_blip_constrained_16 --eps 16 --alpha 1
  python -u instructblip_visual_attack.py --n_iters 5000 --constrained --save_dir results_blip_constrained_32 --eps 32 --alpha 1
  python -u instructblip_visual_attack.py --n_iters 5000 --constrained --save_dir results_blip_constrained_64 --eps 64 --alpha 1
  python -u instructblip_visual_attack.py --n_iters 5000 --save_dir results_blip_unconstrained --alpha 1
  ```

* Testing on the RealToxicityPrompts Dataset

  Make inference on the dataset:

  ```bash
  python instructblip_inference.py --image_file path_to_the_adversarial_example --output_file result.jsonl
  ```

  The `get_metric.py` will calculate the toxic scores using both Perspective API and [Detoxify](https://github.com/unitaryai/detoxify).

  ```bash
  python get_metric.py --input result.jsonl --output result_eval.jsonl
  ```

  Then, you can run `cal_metrics.py` to summarize the evaluation results from the two evaluation:

  ```bash
  python cal_metrics.py --input result_eval.jsonl
  ```

### LLaVA-LLaMA-2

* Set up the environment following the instructions of the original repository at: https://github.com/haotian-liu/LLaVA

  * After get the model, save it to the path `./ckpts/llava_llama_2_13b_chat_freeze`

* Attack

  ```bash
  python -u llava_llama_v2_visual_attack.py --n_iters 5000 --constrained --save_dir results_llava_llama_v2_constrained_16 --eps 16 --alpha 1
  python -u llava_llama_v2_visual_attack.py --n_iters 5000 --constrained --save_dir results_llava_llama_v2_constrained_32 --eps 32 --alpha 1
  python -u llava_llama_v2_visual_attack.py --n_iters 5000 --constrained --save_dir results_llava_llama_v2_constrained_64 --eps 64 --alpha 1
  python -u llava_llama_v2_visual_attack.py --n_iters 5000 --save_dir results_llava_llama_v2_unconstrained --alpha 1
  ```

* Testing on the RealToxicityPrompts Dataset

  Make inference on the dataset:

  ```bash
  python -u llava_llama_v2_inference.py --image_file path_to_the_adversarial_example --output_file result.jsonl
  ```

  The `get_metric.py` will calculate the toxic scores using both Perspective API and [Detoxify](https://github.com/unitaryai/detoxify).

  ```bash
  python get_metric.py --input result.jsonl --output result_eval.jsonl
  ```

  Then, you can run `cal_metrics.py` to summarize the evaluation results from the two evaluation:

  ```bash
  python cal_metrics.py --input result_eval.jsonl
  ```

<br><br>

## Citation
If you find this useful in your research, please consider citing:

```
@misc{qi2023visual,
      title={Visual Adversarial Examples Jailbreak Aligned Large Language Models}, 
      author={Xiangyu Qi and Kaixuan Huang and Ashwinee Panda and Peter Henderson and Mengdi Wang and Prateek Mittal},
      year={2023},
      eprint={2306.13213},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
