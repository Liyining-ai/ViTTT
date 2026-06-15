# $\text{ViT}^3$ for Image Generation

This folder provides the DiT-based image generation code used in our $\text{ViT}^3$ experiments. The code follows the official [DiT](https://github.com/facebookresearch/DiT) implementation, with the $\text{ViT}^3$ model components integrated for class-conditional ImageNet generation.

## Results and Models

The released checkpoints for different DiT model sizes are available here:

| Model Family | Dataset | Image Resolution | Checkpoints |
| :---: | :---: | :---: | :---: |
| $\text{ViT}^3$-DiT | ImageNet-1K | 256 $\times$ 256 | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/10ca7e31adb54147b526/) |

Please download the desired checkpoint and place it under `./pretrained_models/`, for example:

```bash
mkdir -p pretrained_models
# put the downloaded checkpoint here, e.g.
# pretrained_models/vit3_dit_s2_256.pt
```

## Usage

### Installation

The environment is the same as the official [DiT](https://github.com/facebookresearch/DiT) codebase. You can either use the provided `environment.yml` or follow the DiT installation instructions.

```bash
conda env create -f environment.yml
conda activate DiT
```

Install extra packages if your local environment does not already include them:

```bash
pip install timm diffusers accelerate
```

### Dataset Preparation

Prepare ImageNet in the standard folder format expected by `torchvision.datasets.ImageFolder`:

```text
/path/to/imagenet/train/
  n01440764/
  n01443537/
  ...
```

### Training

To train a $\text{ViT}^3$-DiT-S/2 model on ImageNet 256 $\times$ 256, run:

```bash
torchrun --nnodes=1 --nproc_per_node=<GPU_NUM> train.py \
  --model DiT-S/2 \
  --data-path /data/imagenet/train \
  --image-size 256 \
  --global-batch-size 256
```

You can change `--model` to other supported DiT variants, such as `DiT-B/2`, `DiT-L/2`, or `DiT-XL/2`, if the corresponding configuration and checkpoint are used.

### Inference

To sample images from a trained checkpoint, run:

```bash
python sample.py \
  --model DiT-S/2 \
  --image-size 256 \
  --ckpt ./pretrained_models/vit3_dit_s2_256.pt \
  --cfg-scale 4.0 \
  --num-sampling-steps 250 \
  --seed 0
```

The generated image grid will be saved as `sample.png`.

### Evaluation

Following the official DiT evaluation pipeline, first generate 50K samples with `sample_ddp.py`:

```bash
torchrun --nnodes=1 --nproc_per_node=<GPU_NUM> sample_ddp.py \
  --model DiT-S/2 \
  --image-size 256 \
  --ckpt ./pretrained_models/vit3_dit_s2_256.pt \
  --num-fid-samples 50000 \
  --sample-dir samples \
  --cfg-scale 1.5 \
  --num-sampling-steps 250
```

This will save generated images and an `.npz` sample batch under `samples/`. To compute FID, Inception Score, sFID, Precision, and Recall, use the TensorFlow evaluation suite from [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations):

```bash
python evaluator.py /path/to/VIRTUAL_imagenet256_labeled.npz /path/to/generated_samples.npz
```

Please refer to the guided-diffusion evaluation README for downloading the ImageNet reference batch.

## Citation

If you find this repo helpful, please consider citing us.

```bibtex
@inproceedings{han2026vit3,
  title={ViT$^3$: Unlocking Test-Time Training in Vision},
  author={Han, Dongchen and Li, Yining and Li, Tianyu and Cao, Zixuan and Wang, Ziming and Song, Jun and Cheng, Yu and Zheng, Bo and Huang, Gao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Acknowledgments

This codebase is built upon the official [DiT](https://github.com/facebookresearch/DiT) implementation. We also follow the evaluation protocol from [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations).
