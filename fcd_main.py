import os
import argparse

import torch

from fcd_solver import FCDSolver
from data_loader import get_loader
from torch.backends import cudnn
import evaluate
from supervised_solver import SupervisedSolver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    if config.mode == 'eval_fmask':
        evaluate.test_landsat8_biome_fmask(config)
        return

    solver = FCDSolver(config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.make_psuedo_masks()
        # evaluate.test_landsat8_biome(solver, config)
    elif config.mode == 'visualize':
        # solver.visualize_predictions()
        solver.visualize_translations()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=1, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=10, help='weight for identity loss')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='L8Biome', choices=['L8Biome'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=str, default='best', help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'visualize'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='specify device, e.g. cuda:0 to use GPU 0')
    parser.add_argument('--experiment_name', type=str, default=None)

    # Directories.
    parser.add_argument('--l8biome_image_dir', type=str, default='data/L8Biome', help='path to patch data')
    parser.add_argument('--orig_image_dir', type=str, default='/media/data/landsat8-biome', help='path to complete scenes')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models')
    parser.add_argument('--sample_dir', type=str, default='outputs/samples')
    parser.add_argument('--result_dir', type=str, default='outputs/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()

    if config.experiment_name is not None:
        config.model_save_dir = f'outputs/{config.experiment_name}/models'
        config.sample_dir = f'outputs/{config.experiment_name}/samples'
        config.result_dir = f'outputs/{config.experiment_name}/results'

    config.num_channels = 10 if config.dataset == 'L8Biome' else 3

    print(config)
    main(config)
