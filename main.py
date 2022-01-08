import argparse
import os
import torch
import numpy
import random
from fake_news import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Training models for Fake News Detection')
    parser.add_argument('--gpu', type=str, help='the cuda devices used for training', default="0,1,2,3")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_model()