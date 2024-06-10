#!/bin/bash

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

python train_text_diffusion.py \
    --mixed_precision fp16 \
    --dataset_name roc \
    --learning_rate 1e-4 --num_train_steps 10000 --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --tx_dim 768 --tx_depth 12 \
    --objective pred_x0 --enc_dec_model /home/ckfan/controlled_text_generation/bart-base \
    --num_samples 1000 \
    --self_condition \
    --normalize_latent --scale_shift --loss_type l1 --beta_schedule linear \
    --disable_dropout