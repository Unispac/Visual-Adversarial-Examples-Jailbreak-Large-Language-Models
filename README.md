<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Visual Adversarial Examples Jailbreak<br>Large Language Models </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://unispac.github.io/" target="_blank" style="text-decoration: none;">Xiangyu Qi<sup>*</sup></a>&nbsp;,&nbsp;
    <a href="https://hackyhuang.github.io/" target="_blank" style="text-decoration: none;">Kaixuan Huang<sup>*</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=rFC3l6YAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Ashwinee Panda</a><br>
    <a href="https://mwang.princeton.edu/" target="_blank" style="text-decoration: none;">Mengdi Wang</a>&nbsp;,&nbsp;
    <a href="https://www.princeton.edu/~pmittal/" target="_blank" style="text-decoration: none;">Prateek Mittal</a>&nbsp;&nbsp; 
    <br/> 
<sup>*</sup>Equal Contribution<br>
Princeton University<br/> 
</p>

<p align='center';>
<b>
<em>arXiv-Preprint, 2023</em> <br>
</b>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="" target="_blank" style="text-decoration: none;">arXiv</a>&nbsp;
</b>
</p>
-------------------------

#### Abstract

Recently, there has been a surge of interest in introducing vision into Large Language Models (LLMs). The proliferation of large Visual Language Models (VLMs), such as [Flamingo](), [BLIP-2](), and [GPT-4](), signifies an exciting convergence of advancements in both visual and language foundation models. Yet, the risks associated with this integrative approach are largely unexamined. In this paper, we shed light on the security and safety implications of this trend. **First**, we underscore that the continuous and high-dimensional nature of the additional visual input space intrinsically makes it a fertile ground for adversarial attacks. This unavoidably *expands the attack surface of LLMs*. **Second**, we highlight that the broad functionality of LLMs also presents visual attackers with a wider array of achievable adversarial objectives, *extending the implications of security failures* beyond mere misclassification. 

To elucidate these risks, we study adversarial examples in the visual input space of a VLM. <p style="color:red">Specifically, against [MiniGPT-4](), which incorporates safety mechanisms that can refuse harmful instructions, we present **visual adversarial examples** that can circumvent the safety mechanisms and provoke harmful behaviors of the model. Remarkably, we discover that adversarial examples, even if optimized on a narrow, manually curated derogatory corpus against specific social groups, can universally **jailbreak** the model's safety mechanisms. A single such adversarial example can generally undermine MiniGPT-4's safety, enabling it to heed a wide range of harmful instructions and produce harmful content far beyond simply imitating the derogatory corpus used in optimization.</p> Unveiling these risks, we accentuate the urgent need for comprehensive risk assessments, robust defense strategies, and the implementation of responsible practices for the secure and safe utilization of VLMs.

-------



![](assets/demo.png)



In folder `adversarial_images/`, we provide our sample adversarial images under different constraints. Our qualitative results can be verified through the huggingface space https://huggingface.co/spaces/Vision-CAIR/minigpt4.



Code and other materials will be released soon!
