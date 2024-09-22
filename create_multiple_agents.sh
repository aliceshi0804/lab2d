#!/bin/bash

for ((seed=0; seed<8; seed++)); do
    bazel run -c opt dmlab2d/random_agent -- --level_name=chase_eat --num_episodes=1000 --env_seed=$seed
done
