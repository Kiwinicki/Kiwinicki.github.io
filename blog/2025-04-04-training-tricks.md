# Random training tips and tricks

List of all sort of things you can use make your training/finetuning go faster. <!-- truncate --> From one-line tricks to whole architecture changes. Order is random and grouping is somewhat arbitrary. Treat this as a knowledge dump.

1. buy bigger GPU.

## Optimizer

- 8-bit Adam -  2\*8-bit vs 2\*32-bit = ~75% less memory for optimizer states
- Not storing activations from forward pass but recomputing them in backward pass (memory-compute tradeoff)
- "Stateless" optimizers - SGD, Lion. In contrast to e.g. Adam these don't store additional momentum variables (which takes additional 2x model size of space). The catch is these are not drop-in replacements and Adam was specially made for faster convergence & more stable training.  
- Optimizer states offloading to CPU, using [`torchao`](https://github.com/pytorch/ao/tree/main/torchao/optim#optimizer-cpu-offload):
    ```python
    optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
    optim.load_state_dict(ckpt["optim"])
    ```

## PEFT

- LoRA (Low-Rank Adaptation) - thin layer over frozen pretrained layers of your model. It uses a trick of decomposing a large matrix into two smaller low-rank ($n\times m$ where $n << m$) matrices that gives huge memory savings. Slightly more formal:

    $\text{act}[(W + \Delta W)\cdot x + b]$, where:
    - $W=(d \times k)$ pretrained weight tensor
    - $\Delta W = AB$, (init LoRA tensors) where:
        - $A = (d\times \text{rank}) =\mathcal{N}(0, \sigma^2)$ at init
        - $B=(\text{rank} \times k)=0$ at init
        - at the begining $AB$ are no-op but thanks to $A$ being Gaussian there will be symmetry breaking
    - some good links about it: [Sebastian Raschka](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch), [AI Coffe Break](https://youtu.be/KEv-F5UkhxU?si=JcCjdEV-7tXPvk2r)
- QLoRA - It differs in that base model is quantized (usually to 4-bit) but LoRA layers are kept in 16-bit precision
- After training its good to merge LoRA layers with the base model for less overhead (but you can't swap them later).
- DoRA - decomposition of weight matrix into magnitude vector $m$ (euclidean distance) & directional matrix $V$ (angle) and train them separately.
- GaLore - LoRA but for gradient matrix. Supposedly works also for pretraining in contrast to LoRA, but I didn't tried it.
- Prefix Tuning - In place of the prompt you put a random init vector (the so-called prefix) and optimize it until you get the correct answer.
    - "+" tiny amount of parameters to tune
    - "-" takes context length (but alternatively you would put pre-prompt there)
    - "-" interpretability - these are not words, but you can decode it to "nearest" words but it often gibberish

## Training initialization

- init weights properly (relaying on default pytorch init isn't always optimal)
- LR scheduler (OneCycleLR, lr warmup, etc.) and lr search. Similar for BS (batch size warmup etc.)
- max out batch size to fill whole VRAM ([batch size finder](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html))
- some rule regarding relation of LR & BS - [$LR \approx \sqrt{BS}$](https://x.com/cloneofsimo/status/1907731069878825400) (there is also issue of "[critical batch size](https://x.com/SeunghyunSEO7/status/1877188952920125836)")
- if you can't fit enough batch size for stable training (e.g. bsz=1) then use gradient accumulation. In pure pytorch it is calling opt.step() less frequently what results in effectively higher batch size.
- gradient clipping - prevents exploding gradients by capping their magnitude during backpropagation.
- Start from pretrained model (transfer learning) and swap last layer.
    - Freeze whole model except last layer and after few epochs gradually unfreeze rest of the layers.

## Architecture

- normalization layers (stabilize and speeding up training - you can use higher LR)
- turn off bias before BatchNorm (bn already does shift)
- MoE - model architecture which selectively activates only part of the model. memory-computation tradeoff. MoE faster achives same loss under the same computational budged compared to dense models.
- MoD (Mixture of Depths) - learned skip connection for each transformer block, model learns to not waste compute on easy tokens.
- Stochastic Depth - each layer in **deep** ConvNet have probability of not being dropped from 1.0 (for first layer) to 0.5 (for last layer). Simply dropout whole layers (prevents vanishing gradients, faster training,  better performance)

### LLM only

- SuperBPE - groups frequent word sequences into single tokens, improving efficiency and performance. Common word combinations get treated as one unit by the tokenizer, which reduces the number of easy-to-predict sequences. This creates a more balanced prediction difficulty across tokens, allowing the model to distribute computational effort more effectively. [Author explaination](https://x.com/alisawuffles/status/1903125390618661068)

### Diffusion only

- latent diffusion - diffuse in latent space, not pixel space. VAE encoder $\rightarrow$ latent (diffuse $N\times$) $\rightarrow$ VAE decoder. SD1.5 VAE maybe big but you can use [TAESD](https://huggingface.co/madebyollin/taesd) ($(3\cdot512\cdot512)/(16\cdot64\cdot64)=12\times$ compression, 5MB for enc/dec each and minimal computational overhead)
    - if you want train on ImageNet: https://huggingface.co/datasets/fal/cosmos-imagenet (compressed to 2.45GB)
- [Min-SNR](https://arxiv.org/abs/2303.09556) - method of adding a weightning to the loss based on the SNR (signal to noise ratio) of the timestep. It prevents conflicting gradients from different deniosing phases (beggining, mid and final refinements)

## Mixed precision

https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html

https://medium.com/@jbensnyder/solving-the-limits-of-mixed-precision-training-231019128b4b

All types other than FP32 are only faster on consumer cards due to Tensor cores which are only available on RTX cards. Tensor cores are used automatically when using mixed precision.

- **FP32+FP16/BF16** (speed & memory*) - you store an extra copy of the model in 16-bit to be able to do faster calculations (~2x) (memory-compute trade-off). Gradients are also **computed** in 16-bit, but **stored** in FP32. BF16 should be more stable than FP16, as more range is more important for NNs than precision. BF16 stands from "Brain Float" from Google Brain btw.
    
    $$
    \begin{align*}
    \underbrace{(4+2\ \text{bytes})}_{\text{{model FP32+FP16}}} + 
    \underbrace{(2\ \text{bytes})}_{\text{{activation FP16}}} + \underbrace{(4\ \text{bytes})}_{\text{{gradients FP32}}} = 12\ \text{bytes (+8 bytes for Adam)}\end{align*}
    $$
    
- turn on **TF32** (Tensor Float) (speed) - not all operations are supported in 16-bit mixed-precision and have to be done in FP32. Turning on TF32 replaces FP32 in computation (storage still in FP32) at speeds similar to FP16. TF32 is supported on Ampere arch and newer. Fun fact is that TF32 is 19-bit format but has "32" in name.
    
    ```python
    # The flag below controls whether to allow TF32 on matmul.
    # This flag defaults to False in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # The flag below controls whether to allow TF32 on cuDNN. 
    # This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    ```
    
- **FP32+FP8** (speed & memory) - there is possibility to do computations in 8-bit with FP32 accumulation but its more complicated to setup than AMP natively supported in PyTorch. Use [HF Accelerate](https://huggingface.co/docs/accelerate/usage_guides/low_precision_training) library to do this (from what I understand its a wrapper on 3 other packages [`TransformersEngine`, `MS-AMP`, and `torchao`] but not only this).
    
    $$
    \begin{align*}
    \underbrace{(1\ \text{byte})}_{\text{{model FP8 E4M3}}} + 
    \underbrace{(1\ \text{byte})}_{\text{{activation FP8 E4M3}}} + 
    \underbrace{(1\ \text{byte})}_{\text{{gradients FP8 E5M2}}} = 3\ \text{bytes (+8 bytes for Adam)}\end{align*}
    $$

## Other

- avoid moving tensors to another device `.to(device)`, create tensors directly on target device instead. If you don't have any synchronization later in code then you can use `.to(non_blocking=True)`
- use `torch.compile()` if it works, for me it usualy don't.
- set gradients to `None` instead of default `0` (but this can cause some unexpected behaviors - `None` isn't a number so operations with it produces `NaN`)
- use `.as_tensor()` rather than `.tensor()`. `torch.tensor()` always copies data. If you have a numpy array that you want to convert, use `torch.as_tensor()` or `torch.from_numpy()` to avoid copying the data.
- try "channel last" format for tensors and model (NCHW => NHWC), sometimes it's faster. [link](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
- `torch.backends.cudnn.benchmark = True`
- slow dataloader optimizations [Simo tweet](https://x.com/cloneofsimo/status/1855608988681080910), [PyTorch forum](https://discuss.pytorch.org/t/how-to-prefetch-data-when-processing-with-gpu/548/19)
- non-global Cross-Entropy calculation reduces memory usage spike at the end (especially beneficial for LLMs). [paper](https://arxiv.org/abs/2411.09009)
- checkpoint averaging - weighted average of previous checkpoints makes loss landscape more smooth and convex which speeds up training + reduces overfitting (applies to pretraining & finetuning)