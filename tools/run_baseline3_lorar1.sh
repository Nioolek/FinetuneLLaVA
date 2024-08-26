#!/bin/bash

# 4卡

MODELBASE=liuhaotian/llava-v1.5-7b
SAVENAME=baseline3_lorar1

NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 deepspeed finetune.py \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --deepspeed zero2.json \
    --model_name_or_path $MODELBASE  \
    --version v1 \
    --data_path /root/autodl-tmp/data/logo0821/json/train_0821_update.json,/root/autodl-tmp/data/logo0821/json/train_none.json \
    --image_folder /root/autodl-tmp/data/logo0821/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$SAVENAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

python merge_lora.py --model-base $MODELBASE --model-path checkpoints/$SAVENAME --save-model-path merge/$SAVENAME

python test_logo.py --model-path merge/$SAVENAME --result-path result/$SAVENAME.csv