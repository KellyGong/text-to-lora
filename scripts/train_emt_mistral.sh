WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train_custom_sft.py \
    configs/hyper_lora_decontam_lol_tasks.yaml \
    --model_dir=google/gemma-2-2b-it \
    --warmup_frac=0.2 --lr=2.5e-5 --n_tasks_per_batch=8 \
    --n_points_per_task=1 --grad_accum_steps=1 \
    --epochs=10000 --n_descs_per_ds=128 --n_train_ds=64 \
    --exp_setup=emt_lora --encoder_type=linear \
    --l2_reg_generated_w=1e-3 --label_smoothing=0.1 \
    --neftune_noise_alpha=5 --weight_decay=1e-2 --val_batch_size=8
