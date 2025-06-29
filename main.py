import os
import torch
import argparse
from torch.backends import cudnn
from models.FMID import build_net
from train import _train
from test import _test
import numpy as np
import random


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()
    print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _test(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='FMID',type=str)
    parser.add_argument('--data_dir', type=str, default=r'E:\mydatas\MYRESIDEOTS')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)


    # Train
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')




    # Test
    parser.add_argument('--test_model', type=str, default=r'W:\DB\FMID-main\checkpoint\reside.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', 'FMID', 'sots/')
    args.result_dir = os.path.join('results/', args.model_name, 'test')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    command = 'cp ' + 'models/layers.py' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/FMID.py' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    print(args)
    main(args)
