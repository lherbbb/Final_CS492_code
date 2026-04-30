export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="./logs/OSCP"

accelerate launch train_lora.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --lora_rank=64 \
    --lmd=0.001 \
    --N=2 \
    --learning_rate=1e-4 --loss_type="l2" --adam_weight_decay=0.0 \
    --max_train_steps=10000 \
    --max_train_samples=40000 \
    --dataloader_num_workers=8 \
    --checkpointing_steps=2000 --checkpoints_total_limit=10 \
    --train_batch_size=16 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --lr_scheduler="constant_with_warmup" \
    --resume_from_checkpoint=latest \
    --report_to="tensorboard" \
    --seed=3407