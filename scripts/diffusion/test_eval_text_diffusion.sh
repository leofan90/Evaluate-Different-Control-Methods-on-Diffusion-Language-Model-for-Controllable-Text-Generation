#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python train_text_diffusion.py \
    --eval_test \
    --resume_dir saved_models/ag_news/epochs_10000 \
    --sampling_timesteps 250 --num_samples 1000 --ddim_sampling_eta 1