#!/usr/bin/env bash
PRETRAIN_DIR=/tmp/models/imdb_pretrain
IMDB_DATA_DIR=/tmp/imdb
TRAIN_DIR=/tmp/models/imdb_classify


python gen_data.py \
    --output_dir=/home/paperspace/imdb_es_records \
    --dataset=imdb \
    --imdb_input_dir=/home/paperspace/imdb_es \
    --lowercase=False \
    --label_gain=False
