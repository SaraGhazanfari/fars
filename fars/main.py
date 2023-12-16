import argparse
import os
import sys
import warnings
from os.path import realpath

from fars.eval_linear import LinearEvaluation

warnings.filterwarnings("ignore")


def override_args(config, depth, num_channels, depth_linear, n_features):
    config.depth = depth
    config.num_channels = num_channels
    config.depth_linear = depth_linear
    config.n_features = n_features
    return config


def set_config(config):
    if config.model_name == 'small':
        config = override_args(config, 20, 45, 7, 1024)  # depth, num_channels, depth_linear, n_features
    elif config.model_name == 'medium':
        config = override_args(config, 30, 60, 10, 2048)
    elif config.model_name == 'large':
        config = override_args(config, 50, 90, 10, 2048)
    elif config.model_name == 'xlarge':
        config = override_args(config, 70, 120, 15, 2048)
    elif config.model_name is None and \
            not all([config.depth, config.num_channels, config.depth_linear, config.n_features]):
        ValueError("Choose --model-name 'small' 'medium' 'large' 'xlarge'")

    # cluster constraint

    # process argments
    os.makedirs('./trained_models', exist_ok=True)
    path = realpath('./trained_models')
    config.train_dir = f'{path}/{config.train_dir}'

    return config


def main(config):
    config = set_config(config)
    linear_eval = LinearEvaluation(config)
    linear_eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Evaluate Lipschitz Networks.')
    parser.add_argument("--train_dir", default=None, type=str, help="Name of the training directory.")
    parser.add_argument("--data_dir", type=str, help="Name of the data directory.")
    parser.add_argument("--dataset", type=str, default='imagenet', help="Dataset to use")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--loss", type=str, default="rmse", choices=['rmse', 'hinge', 'cross', 'byol'],
                        help="Define the loss to use for training.")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--print_grad_norm", action='store_true', help="Print of the norm of the gradients")
    parser.add_argument("--frequency_log_steps", type=int, default=100, help="Print log for every step.")
    parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint every epoch.")

    # specific parameters for eval
    # parser.add_argument("--attack", type=str,
    #                     choices=['PGD-L2', 'PGD-Linf', 'PGD-L1', 'AA-L2', 'AA-Linf', 'CW-L2', 'CW-Linf', 'SQ-Linf',
    #                              'SQ-L2', 'DF-L2', 'MI-L2', 'MI-Linf'],
    #                     help="Choose the attack.")
    # parser.add_argument("--eps", type=float, default=36)

    # parameters of the architectures
    parser.add_argument("--model-name", type=str, default='small')
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--num_channels", type=int, default=30)
    parser.add_argument("--depth_linear", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=2048)
    parser.add_argument("--conv_size", type=int, default=5)
    parser.add_argument("--init", type=str, default='xavier_normal')

    parser.add_argument("--num_linear", type=int, default=1)
    parser.add_argument("--simplex", type=str, default='argmax')

    # parse all arguments
    config = parser.parse_args()
    config.cmd = f"python3 {' '.join(sys.argv)}"

    main(config)
