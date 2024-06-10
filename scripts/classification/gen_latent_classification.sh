#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python train_latent_classifier.py \
    --gen_data \
    --resume_classifier_dir saved_models/ag_news/epochs_10000_both \
    --sampling_timesteps 250 \
    --num_samples 1000 --ddim_sampling_eta 1 \