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

    solver = SupervisedSolver(config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        evaluate.test_landsat8_biome_supervised(solver, config)
    elif config.mode == 'visualize':
        solver.visualize_preds()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--dataset', type=str, default='L8Biome', choices=['L8Biome'])
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cam_threshold', type=float, default=0.10)
    parser.add_argument('--cam_method', type=str, choices=['cam', 'gradcam', 'gradcampp', 'ucam'], default='cam')

    # Test configuration.
    parser.add_argument('--test_iters', type=str, default='best', help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_cam', 'test_cam', 'visualize', 'make_cam'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='specify device, e.g. cuda:0 to use GPU 0')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--debug', action='store_true')

    # Directories.
    parser.add_argument('--l8biome_image_dir', type=str, default='data/L8Biome')
    parser.add_argument('--l8sparcs_image_dir', type=str, default='/media/data/SPARCS')
    parser.add_argument('--orig_image_dir', type=str, default='/media/data/landsat8-biome', help='path to complete scenes')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models')
    parser.add_argument('--sample_dir', type=str, default='outputs/samples')
    parser.add_argument('--result_dir', type=str, default='outputs/results')

    parser.add_argument('--train_mask_file', type=str, default='mask.tif')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='fraction of training data to use')
    parser.add_argument('--encoder_weights', type=str, default=None)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--train_encoder_only', type=str2bool, default=False)
    parser.add_argument('--freeze_encoder', type=str2bool, default=False)
    parser.add_argument('--classifier_head', type=str2bool, default=False)
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()

    if config.experiment_name is not None:
        config.model_save_dir = f'outputs/{config.experiment_name}/models'
        config.sample_dir = f'outputs/{config.experiment_name}/samples'
        config.result_dir = f'outputs/{config.experiment_name}/results'

    config.num_channels = 10 if config.dataset == 'L8Biome' else 3
    if config.train_encoder_only:
        config.train_mask_file = None

    print(config)
    main(config)
