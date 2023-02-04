[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/input-perturbation-reduces-exposure-bias-in/image-generation-on-celeba-64x64)](https://paperswithcode.com/sota/image-generation-on-celeba-64x64?p=input-perturbation-reduces-exposure-bias-in)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/input-perturbation-reduces-exposure-bias-in/image-generation-on-imagenet32)](https://paperswithcode.com/sota/image-generation-on-imagenet32?p=input-perturbation-reduces-exposure-bias-in)


## DDPM-IP
This is the codebase for the paper [Input Perturbation Reduces Exposure Bias in Diffusion Models](https://arxiv.org/abs/2301.11706).

This repository is heavily based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with training modification of input perturbation.


## Simple to implement Input Perturbation in DDPMs
Our proposed __Input Perturbation__  is an extremely simple __plug-in__ method for general diffusion models.
The implementation of Input Perturbation is just two lines of code.
For instance, based on [guided-diffusion](https://github.com/openai/guided-diffusion),
the only code modifications are in the script `guided_diffusion/gaussian_diffusion.py`:

```python
new_noise = noise + garmma * th.randn_like(noise)
x_t = self.q_sample(x_start, t, noise=new_noise)
```

NOTE THAT: change the parameter `GPUS_PER_NODE = 4` in the script `dist_util.py` according to your GPU cluster configuration.


## Installation
the installation is the same with [guided-diffusion](https://github.com/openai/guided-diffusion)
```
git clone https://github.com/forever208/DDPM-IP.git
cd DDPM-IP
pip install -e .
```


## Download ADM-IP pre-trained models

We have released checkpoints for the main models in the paper.

Here are the download links for each model checkpoint:

 * CIFAR10 32x32: [ADM-IP.pt](https://drive.google.com/file/d/1-3WpnfIZL7VNDK4GBBQ2IvJCPxoEaGwD/view?usp=share_link)
 * ImageNet 32x32: [ADM-IP.pt](https://drive.google.com/file/d/1FFUJDk-__9y9DnAG6DKDx5W7LgEIuJyk/view?usp=share_link)
 * LSUN tower 64x64: [ADM-IP.pt](https://drive.google.com/file/d/1QUaY94bSAiTdGu5T_GtdIXMvGT3X1GMy/view?usp=sharing)
 * CelebA 64x64: [ADM-IP.pt](https://drive.google.com/file/d/1Us9zKaIMh8dDlAZVXt3hR2FQYwhxEPYk/view?usp=sharing)
 

## Sampling from pre-trained ADM-IP models

To unconditionally sample from these models, you can use the `image_sample.py` scripts.
Sampling from DDPM-IP has no difference with sampling from `openai/guided-diffusion` since DDPM-IP does not change the sampling process.

For example, we sample 50k images from CIFAR10 by: 
```
mpirun python scripts/image_sample.py \
--image_size 32 --timestep_respacing 100 \
--model_path PATH_TO_CHECKPOINT \
--num_channels 128 --num_head_channels 32 --num_res_blocks 3 --attention_resolutions 16,8 \
--resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.3 \
--diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True --batch_size 256 --num_samples 50000
```


## Results

This table summarizes our input perturbation results based on ADM baselines.

FID computation details: 
- All FIDs are computed using 50K generated samples (unconditional sampling). 
- For CIFAR10 and ImageNet 32x32, we use the whole training data as the reference batch, 
- For LSUN tower 64x64 and CelebA 64x64, we randomly pick up 50k samples from the training set, forming the reference batch   

<p align="left">
  <img src="https://github.com/forever208/DDPM-IP/blob/DDPM-IP/datasets/DDPM-IP-results.png" width='100%' height='100%'/>
</p>



## Training ADM-IP

Training diffusion models is described in this [repository](https://github.com/openai/improved-diffusion).

Training DDPM-IP only requires one more argument `--input perturbation 0.1`.

(set `--input perturbation 0.0` for the baseline DDPM)

We share the complete arguments of training in the four datasets:


CIFAR10
```
mpiexec -n 2  python scripts/image_train.py --input_pertub 0.1 \
--data_dir PATH_TO_DATASET \
--image_size 32 --use_fp16 True --num_channels 128 --num_head_channels 32 --num_res_blocks 3 \
--attention_resolutions 16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 64
```

ImageNet 32x32 (you can also choose dropout=0.1)
```
mpiexec -n 4  python scripts/image_train.py --input_pertub 0.1 \
--data_dir PATH_TO_DATASET \
--image_size 32 --use_fp16 True --num_channels 128 --num_head_channels 32 --num_res_blocks 3 \
--attention_resolutions 16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 128
```

LSUN tower 64x64
```
mpiexec -n 16  python scripts/image_train.py --input_pertub 0.1 \
--data_dir PATH_TO_DATASET \
--image_size 64 --use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 16
```

CelebA 64x64
```
mpiexec -n 16  python scripts/image_train.py --input_pertub 0.1 \
--data_dir PATH_TO_DATASET \
--image_size 64 --use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 16
```

