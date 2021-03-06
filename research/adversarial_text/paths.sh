#!/usr/bin/env bash
PRETRAIN_DIR=/tmp/models/imdb_pretrain
IMDB_DATA_DIR=/tmp/imdb
TRAIN_DIR=/tmp/models/imdb_classify


python gen_data.py \
    --output_dir=/tmp/imdb_small_records \
    --dataset=imdb \
    --imdb_input_dir=/tmp/imdb_small \
    --lowercase=False \
    --label_gain=False


python gen_data.py \
    --output_dir=/home/paperspace/imdb_small_records \
    --dataset=imdb \
    --imdb_input_dir=/home/paperspace/imdb_es \
    --lowercase=False \
    --label_gain=False

# note vocab size change in below

python pretrain.py \
    --train_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --num_candidate_samples=1024 \
    --batch_size=256 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.9999 \
    --max_steps=100000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings


TRAIN_DIR=/tmp/models/imdb_classify
python train_classifier.py \
    --train_dir=$TRAIN_DIR \
    --pretrained_model_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --cl_num_layers=1 \
    --cl_hidden_size=30 \
    --batch_size=64 \
    --learning_rate=0.0005 \
    --learning_rate_decay_factor=0.9998 \
    --max_steps=15000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings \
    --adv_training_method=vat \
    --perturb_norm_length=5.0



EVAL_DIR=/tmp/models/imdb_eval
python evaluate.py \
    --eval_dir=$EVAL_DIR \
    --checkpoint_dir=$TRAIN_DIR \
    --eval_data=test \
    --run_once \
    --num_examples=25000 \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=86934 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --batch_size=256 \
    --num_timesteps=400 \
    --normalize_embeddings
