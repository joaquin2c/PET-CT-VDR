# CT Encoders

This repository contains the code used and modified from the [MAE](https://github.com/facebookresearch/mae) and [iBOT](https://github.com/bytedance/ibot) repositories to pre-train encoders with ViT architecture using computed tomography images.

## Installation

1. Create a virtual environment: 
   ```bash
   conda create -n ctencoders python=3.10 -y
   ```
   and activate it: 
   ```bash
   conda activate ctencoders
   ```
2. Install [PyTorch 2.5](https://pytorch.org/get-started/locally/)
3. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### iBOT Training

For a glimpse at the full documentation of iBOT pre-training, please run:
```bash
python ibot/main_ibot.py --help
   ```
To start the iBOT pre-training with Vision Transformer (ViT), simply run the following command:
```bash
python ibot/main_ibot.py --data_path /../your_dataset --output_dir ./output_dir
   ```
To start the iBOT pre-training with Swin3D, simply run the following command:
```bash
python ibot/main_ibot.py \
--arch swin3D \
--batch_size_per_gpu 2 \
--epochs 400 \
--data_path /../your_dataset \
--saveckp_freq 40 \
--clip_grad 3.0 \
--global_crops_scale 0.4 1.0 \
--local_crops_scale 0.05 0.4 \
--lr 0.0005 \
--min_lr 1e-06 \
--norm_last_layer False \
--patch_size 4 \
--pred_ratio 0.0 0.3 \
--pred_ratio_var 0.0 0.2 \
--pred_shape block \
--pred_start_epoch 50 \
--warmup_teacher_temp_epochs 30 \
--window_size 7 \
--output_dir ./output_dir
   ```

### MAE Training

For a glimpse at the full documentation of MAE pre-training, please run:
```bash
python mae/main_pretrain.py --help
   ```
To start the MAE pre-training, simply run the following command:
```bash
python mae/main_pretrain.py --data_path /../your_dataset --output_dir ./output_dir
   ```

   





