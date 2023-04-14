#!/bin/bash

python run.py --learning_rate 1e-6 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 5e-6 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 1e-5 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 5e-5 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 1e-4 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 5e-4 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 1e-3 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 5e-3 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 1e-2 --devices 3 --epoch 30 --patience 10 --batch_size 32
python run.py --learning_rate 5e-2 --devices 3 --epoch 30 --patience 10 --batch_size 32
