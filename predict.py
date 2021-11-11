import os
import pickle as pkl

import pandas as pd
import torch
from tqdm import tqdm
import yaml

from features.data import construct_loader
from model.model import ReactionModel
from utils.parsing import parse_command_line_arguments
from utils.utils import create_logger

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
test_loader = construct_loader(args, modes=('test'))

# load stdzer
with open(args.sdtzer_path, "rb") as f:
    stdzer = pkl.load(f)

# load weights
logger.info(f'Reading in model weights from {args.state_dict}')
with open(args.model_params, 'r') as f:
    model_params = yaml.load(stream=f, Loader=yaml.UnsafeLoader)
model = ReactionModel(**model_params).to(device)
model.load_state_dict(torch.load(args.state_dict, map_location=device))
model.eval()

preds = []
for batch in tqdm(test_loader):
    batch = batch.to(device)
    out = model(batch.ts_z, batch.ts_coords, batch.r_z, batch.r_coords, batch.r_z_batch, batch.ffn_inputs)
    preds.extend(stdzer(out, rev=True).detach().cpu().numpy().flatten().tolist())

# save preds to a csv
df = pd.DataFrame(preds, columns=['predictions'])
df.to_csv('predictions.csv', index=False)
