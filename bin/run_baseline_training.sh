#!/bin/bash

# training and validation for turn detection
model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base
cuda_id=0

CUDA_VISIBLE_DEVICES=${cuda_id} python baseline.py \
        --task detection \
        --dataroot data \
        --model_name_or_path ${model_name} \
        --params_file baseline/configs/detection/params.json \
        --exp_name td-review-${model_name_exp}-baseline \
        --knowledge_file knowledge.json


# training and validation for knowledge selection
model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base
cuda_id=1

CUDA_VISIBLE_DEVICES=${cuda_id} python baseline.py \
        --task selection \
        --dataroot data \
        --model_name_or_path ${model_name} \
        --negative_sample_method "oracle" \
        --knowledge_file knowledge.json \
        --params_file baseline/configs/selection/params.json \
        --exp_name ks-review-${model_name_exp}-oracle-baseline


# training and validation for response generation
model_name=facebook/bart-base
model_name_exp=bart-base
cuda_id=2

CUDA_VISIBLE_DEVICES=${cuda_id} python baseline.py \
        --params_file baseline/configs/generation/params.json \
        --task generation \
        --dataroot data \
        --model_name_or_path ${model_name} \
        --history_max_tokens 256 --knowledge_max_tokens 256 \
        --knowledge_file knowledge.json \
        --exp_name rg-review-${model_name_exp}-baseline
