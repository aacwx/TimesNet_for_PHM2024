#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --model_id PHM_TimesNet \
  --model TimesNet \
  --data phm \
  --root_path ./dataset/phm/ \
  --features M \
  --seq_len 1024 \
  --label_len 48 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 8 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 2048 \
  --dropout 0.1 \
  --embed timeF \
  --activation gelu \
  --use_gpu True \
  --gpu 0 \
  --gpu_type cuda \
  --train_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --patience 5 \
  --des PHM_classification \
  --loss CE \
  --itr 1