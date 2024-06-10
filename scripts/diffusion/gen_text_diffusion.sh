#!/bin/bash

python train_text_diffusion.py \
    --gen_data \
    --resume_dir saved_models/humor_speech/epochs_10000_both \
    --sampling_timesteps 250 \
    --num_samples 1000 --ddim_sampling_eta 1