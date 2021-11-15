import argparse
import os

import matplotlib
import torch
from torch.backends import cudnn

import evaluate
from supervised_solver import SupervisedSolver
from cam_solver import CAMSolver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True
    if not config.debug:
        matplotlib.use('pdf')  # disable display

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    solver = CAMSolver(config)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.eval_cam()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Train configuration.
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--dataset', type=str, default='L8Biome', choices=['L8Biome'])
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cam_threshold', type=float, default=0.10)
    parser.add_argument('--cam_method', type=str, choices=['cam', 'gradcam', 'gradcampp', 'ucam'], default='cam')
    parser.add_argument('--num_channels', type=int, default=10)
    parser.add_argument('--pretrained', type=str2bool, default=True, help='whether to load imagenet weights')

    # Test configuration.
    parser.add_argument('--test_checkpoint', type=str, default='best', help='test model from this checkpoint')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='specify device, e.g. cuda:0 to use GPU 0')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--debug', action='store_true', help='disable matplotlib')

    # Directories.
    parser.add_argument('--l8biome_image_dir', type=str, default='data/L8Biome')
    parser.add_argument('--l8sparcs_image_dir', type=str, default='/media/data/SPARCS')
    parser.add_argument('--orig_image_dir', type=str, default='/media/data/landsat8-biome', help='path to complete scenes')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models')
    parser.add_argument('--sample_dir', type=str, default='outputs/samples')
    parser.add_argument('--result_dir', type=str, default='outputs/results')
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()

    if config.experiment_name is not None:
        config.model_save_dir = f'outputs/{config.experiment_name}/models'
        config.sample_dir = f'outputs/{config.experiment_name}/samples'
        config.result_dir = f'outputs/{config.experiment_name}/results'

    print(config)
    main(config)
