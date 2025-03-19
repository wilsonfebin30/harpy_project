#!/bin/bash

python scripts/rsl_rl/train.py --task=Harpy-Flat-Train --num_envs 512 --headless --video --video_length 800 --video_interval 4000 --enable_cameras
