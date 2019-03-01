#!/usr/bin/env bash

$ wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
    -O /tmp/imdb.tar.gz
tar -xf /tmp/imdb.tar.gz -C /tmp
PRETRAIN_DIR=/tmp/models/imdb_small_pretrain
IMDB_DATA_DIR=/tmp/imdb_small_records



python gen_data.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/imdb_small \
    --lowercase=False \
    --label_gain=False


#python gen_data.py \
#    --output_dir=/home/paperspace/imdb_small_records \
#    --dataset=imdb \
#    --imdb_input_dir=/home/paperspace/imdb_es \
#    --lowercase=False \
#    --label_gain=False

# note vocab size change in below

python pretrain.py \
    --train_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --num_candidate_samples=1024 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.9999 \
    --max_steps=30 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings


TRAIN_DIR=/tmp/models/imdb_classify_small
echo $PRETRAIN_DIR
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
    --max_steps=50 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings \
    --adv_training_method=vat \
    --perturb_norm_length=5.0


EVAL_DIR=/tmp/models/imdb_eval_small
python evaluate.py \
    --eval_dir=$EVAL_DIR \
    --checkpoint_dir=$TRAIN_DIR \
    --eval_data=test \
    --run_once \
    --num_examples=10000 \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --batch_size=256 \
    --num_timesteps=400 \
    --normalize_embeddings
