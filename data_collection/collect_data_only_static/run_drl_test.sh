#!/bin/bash

DOMAIN=drl_env

SEED=2
SAVEDIR=../log/DRL/SAC/test

python train_drlenv_test.py \
    --domain_name ${DOMAIN} \
    --agent 'ambs' \
    --init_steps 100 \
    --num_train_steps 10000000 \
    --encoder_type pixel \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --hidden_dim 256 \
    --encoder_feature_dim 100 \
    --num_layers 2 \
    --num_filters 16 \
    --batch_size 1024 \
    --encoder_lr 5e-4 \
    --decoder_lr 5e-4 \
    --actor_lr 5e-4 \
    --critic_lr 5e-4 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --init_temperature 0.01 \
    --num_eval_episodes 0 \
    --work_dir ${SAVEDIR}/new_sac_test_${DOMAIN}_${SEED}_SAC_dy_8_agent_$(date +"%Y-%m-%d-%H-%M-%S") \
    --save_tb \
    --save_model \
    --load_encoder './model/static24_in/actor_1842130.pt' \
    --preference 'local' \
    --output_path 'local_maze.pkl' \
    --env_config 'envs/cfg/circle.yaml' \
    --save_episode 1000 \
    --seed ${SEED} $@ \
