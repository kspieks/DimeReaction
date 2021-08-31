from datetime import datetime
import os
import math
import random

import numpy as np
import torch
import yaml

from features.common import Standardizer
from features.data import construct_loader
from model.model import ReactionModel
from model.nn_utils import get_activation_function, get_optimizer_and_scheduler, NoamLR, param_count
from model.training import train, test
from utils.parsing import parse_command_line_arguments
from utils.utils import (create_logger,
                         dict_to_str,
                         plot_train_val_loss,
                         plot_lr,
                         plot_gnorm_pnorm,
                         save_yaml_file,
                         )

args = parse_command_line_arguments()
log_dir = args.log_dir
logger = create_logger(args.log_name, log_dir)
logger.info('Using arguments...')
for arg in vars(args):
    logger.info(f'{arg}: {getattr(args, arg)}')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'device: {device}')

# construct loader
train_loader, val_loader, test_loader = construct_loader(args, modes=('train', 'val', 'test'))
mean = torch.tensor(train_loader.dataset.mean, dtype=torch.float).to(device)
std = torch.tensor(train_loader.dataset.std, dtype=torch.float).to(device)
stdzer = Standardizer(mean, std)
logger.info(f'\nTraining target mean +- 1 std: {mean} +- {std} kcal/mol')
logger.info(f'Validation target mean +- 1 std: {val_loader.dataset.mean} +- {val_loader.dataset.std} kcal/mol')
logger.info(f'Testing target mean +- 1 std: {test_loader.dataset.mean} +- {test_loader.dataset.std} kcal/mol\n')


# build model
# if fine-tuning, load in previous weights
if args.model_params and args.state_dict:
    logger.info(f'Reading in model weights from {args.state_dict}')
    with open(args.model_params, 'r') as f:
        model_params = yaml.load(stream=f, Loader=yaml.UnsafeLoader)
    model = ReactionModel(**model_params).to(device)
    model.load_state_dict(torch.load(args.state_dict, map_location=device))
# otherwise, create a fresh model
else:
    model_params = {'hidden_channels': args.hidden_channels,
                    'out_emb_channels': args.out_emb_channels,
                    'out_channels': args.out_channels,
                    'int_emb_size': args.int_emb_size,
                    'basis_emb_size': args.basis_emb_size,
                    'num_blocks': args.num_blocks,
                    'num_spherical': args.num_spherical,
                    'num_radial': args.num_radial,
                    'num_output_layers': args.num_output_layers,
                    'cutoff': args.cutoff,
                    'envelope_exponent': args.envelope_exponent,
                    'num_before_skip': args.num_before_skip,
                    'num_after_skip': args.num_after_skip,
                    'activation': get_activation_function(args.activation),
                    # MLP
                    'num_additional_ffn_inputs': len(args.ffn_inputs),
                    'ffn_hidden_size': args.ffn_hidden_size,
                    'out_dim': len(args.targets),
                    'ffn_num_layers': args.ffn_num_layers,
                    'ffn_activation': get_activation_function(args.ffn_activation),
                    'dropout': args.dropout,
                    'layer_norm': args.layer_norm,
                    'batch_norm': args.batch_norm,
                    }
model = ReactionModel(**model_params).to(device)
logger.info(f'Model architecture is:\n{model}')
logger.info(f'Total number of parameters: {param_count(model):,}')

# get optimizer and scheduler and define loss
optimizer, scheduler = get_optimizer_and_scheduler(args, model, len(train_loader.dataset))
loss = torch.nn.MSELoss(reduction='sum')

# record parameters
logger.info(f'\nModel parameters are:\n{dict_to_str(model_params)}\n')
yaml_file_name = os.path.join(log_dir, 'model_params.yml')
save_yaml_file(yaml_file_name, model_params)
logger.info(f'Optimizer parameters are:\n{optimizer}\n')
logger.info(f'Scheduler state dict is:')
if scheduler:
    for key, value in scheduler.state_dict().items():
        logger.info(f'{key}: {value}')
    logger.info('')
logger.info(f'Steps per epoch: {len(train_loader)}')

best_val_rmse = math.inf
best_epoch = 0
logger.info("Starting training...\n")
for epoch in range(1, args.num_epochs):
    train_rmse, train_mae = train(model, train_loader, optimizer, loss, device, scheduler, logger if args.verbose else None, stdzer)
    logger.info(f'Epoch {epoch}: Overall Training RMSE/MAE {train_rmse.mean():.5f}/{train_mae.mean():.5f}')
    for target, rmse, mae in zip(args.targets, train_rmse, train_mae):
        logger.info(f'Epoch {epoch}: {target} Training RMSE/MAE {rmse:.5f}/{mae:.5f}')

    val_rmse, val_mae = test(model, val_loader, device, stdzer)
    logger.info(f'Epoch {epoch}: Overall Validation RMSE/MAE {val_rmse.mean():.5f}/{val_mae.mean():.5f}')
    for target, rmse, mae in zip(args.targets, val_rmse, val_mae):
        logger.info(f'Epoch {epoch}: {target} Validation RMSE/MAE {rmse:.5f}/{mae:.5f}')
    if scheduler and not isinstance(scheduler, NoamLR):
        scheduler.step(val_rmse)

    if val_rmse.mean() <= best_val_rmse:
        best_val_rmse = val_rmse.mean()
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

logger.info(f'\nBest Overall Validation RMSE {best_val_rmse:.5f} on Epoch {best_epoch}')

# create new instance for testing
model = ReactionModel(**model_params).to(device)
model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pt'), map_location=device))
test_rmse, test_mae = test(model, test_loader, device, stdzer)
logger.info(f'When using the model from Epoch {best_epoch}: '
            f'Overall Testing RMSE/MAE {test_rmse.mean():.5f}/{test_mae.mean():.5f}')
for target, rmse, mae in zip(args.targets, test_rmse, test_mae):
    logger.info(f'Epoch {epoch}: {target} Testing RMSE/MAE {rmse:.5f}/{mae:.5f}')

# make plots
log_file = os.path.join(log_dir, args.log_name + '.log')
plot_train_val_loss(log_file)
if args.verbose:
    plot_gnorm_pnorm(log_file)
    plot_lr(log_file)
