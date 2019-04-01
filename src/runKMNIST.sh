#!/bin/sh

python3 main.py --device cpu --run bohb --dataset KMNIST --data_dir ../data -v INFO -o ../results/ --n_iterations 20 --eta 2 --min_budget 1 --max_budget 10 -w 1
