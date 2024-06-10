#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train_latent_classifier.py \
    --mixed_precision fp16 \
    --learning_rate 3e-5 --adam_weight_decay 0.01 \
    --dataset_name humor_speech \
    --num_train_steps 5000 --train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --tx_dim 768 --tx_depth 12 \
    --objective pred_x0 \
    --enc_dec_model /home/ckfan/controlled_text_generation/bart-base \
    --normalize_latent --scale_shift \
    --beta_schedule linear --loss_type l1 \
    --lr_schedule "inverse_sqrt" \
    --resume_diffusion_dir saved_models/humor_speech/epochs_10000_no_condition \
    # --class_conditional \
    # --resume_classifier_dir saved_models/humor_speech/epochs_10000  \
    