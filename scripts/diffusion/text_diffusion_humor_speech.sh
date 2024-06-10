export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=1

python train_text_diffusion.py \
    --learning_rate 0.0001 \
    --dataset_name humor_speech \
    --num_train_steps 10000 --train_batch_size 4 \
    --tx_dim 768 --tx_depth 12 \
    --objective pred_x0 --enc_dec_model /home/ckfan/controlled_text_generation/bart-base \
    --num_samples 1000 \
    --normalize_latent --scale_shift --disable_dropout --beta_schedule linear --loss_type l1 \
    --self_condition # --class_conditional \