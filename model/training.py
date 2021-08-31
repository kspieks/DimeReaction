import math

import torch
import torch.nn as nn
from tqdm import tqdm

from .nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model, loader, optimizer, loss, device, scheduler, logger, stdzer):
    """
    Function used for training the model

    Args:
        model: model to be trained
        loader: instance of torch geometric data loader
        optimizer: a PyTorch optimizer
        loss: a PyTorch loss function
        device: the device
        scheduler: optional learning rate scheduler
        logger: the logger object
        stdzer: standardizes target variables
    """

    model.train()
    rmse_total, mae_total = 0, 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.ts_z, batch.ts_coords, batch.r_z, batch.r_coords, batch.r_z_batch, batch.ffn_inputs)
        result = loss(out, stdzer(batch.y))
        result.backward()

        # clip the gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
            lrs = scheduler.get_lr()
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
        else:
            for param_group in optimizer.param_groups:
                logger.info(param_group['lr'])
            lrs_str = f"lr_0 = {param_group['lr']:.4e}"

        if logger:
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            logger.info(f'Training RMSE: {math.sqrt(result.item()):.5f}, PNorm: {pnorm:.4f}, GNorm: {gnorm:.4f}, {lrs_str}')

        rmse_total += (stdzer(out, rev=True) - batch.y).square().sum(dim=0).detach().cpu()
        mae_total += (stdzer(out, rev=True) - batch.y).abs().sum(dim=0).detach().cpu()

    # divide by number of molecules
    train_rmse_loss = torch.sqrt(rmse_total / len(loader.dataset))
    train_mae_loss = mae_total / len(loader.dataset)

    return train_rmse_loss, train_mae_loss


def test(model, loader, device, stdzer):
    model.eval()
    rmse_total, mae_total = 0, 0

    for batch in tqdm(loader):
        batch = batch.to(device)
        out = model(batch.ts_z, batch.ts_coords, batch.r_z, batch.r_coords, batch.r_z_batch, batch.ffn_inputs)

        rmse_total += (stdzer(out, rev=True) - batch.y).square().sum(dim=0).detach().cpu()
        mae_total += (stdzer(out, rev=True) - batch.y).abs().sum(dim=0).detach().cpu()

    # divide by number of molecules
    val_rmse_loss = torch.sqrt(rmse_total / len(loader.dataset))
    val_mae_loss = mae_total / len(loader.dataset)

    return val_rmse_loss, val_mae_loss
