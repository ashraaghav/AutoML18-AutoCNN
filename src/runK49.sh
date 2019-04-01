#!/bin/sh

# python3 main.py --dataset K49 --data_dir ../data -v INFO -o ../results/k49_full_1/ --n_iterations 25 --eta 3 --min_budget 1 --max_budget 12 -w 1
python3 main.py --device cpu --run bohb --dataset K49 --data_dir ../data -v INFO -o ../results/ --n_iterations 15 --eta 3 --min_budget 1 --max_budget 10 -w 1
