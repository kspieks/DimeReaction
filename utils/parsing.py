from argparse import ArgumentParser


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """
    parser = ArgumentParser()

    parser.add_argument('--log_dir', type=str,
                        help='Directory to store the log file. Datetime is appended.')
    parser.add_argument('--log_name', type=str, default='train',
                        help='Filename for the training log.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Boolean indicating whether to log gradient norm, parameter norm, learning rate, and batch loss.')

    parser.add_argument('--reactant_xyzs', type=str,
                        help='Path to the text file containing xyz coordinates for the reactants.')
    parser.add_argument('--ts_xyzs', type=str,
                        help='Path to the csv file containing xyz coordinates for the transition states.')

    parser.add_argument('--data_path', type=str,
                        help='Path to the csv file containing SMILES, targets, and optional ffn inputs.')
    parser.add_argument('--targets', nargs='+',
                        help='Name of columns to use as regression targets.')
    parser.add_argument('--split_path', type=str,
                        help='Path to .npy file with train, val, and test indices.')

    # optimization parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to run.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use.')
    parser.add_argument('--scheduler', type=str, default='noam',
                        choices=['exponential', 'noam', 'plateau'],
                        help='Learning rate scheduler to use.')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of epochs during which learning rate increases linearly from init_lr to max_lr.'
                             'Afterwards, learning rate decreases exponentially from max_lr to final_lr.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. Max lr when using noam.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for the parallel data loading (0 means sequential).')

    # dimenet++ parameters
    parser.add_argument('--hidden_channels', type=int, default=50,
                        help='Hidden embedding size.')
    parser.add_argument('--out_emb_channels', type=int, default=50,
                        help='Embedding size used for atoms in the output block.')
    parser.add_argument('--out_channels', type=int, default=50,
                        help='Size of each output sample.')
    parser.add_argument('--int_emb_size', type=int, default=64,
                        help='Embedding size used for interaction triplets.')
    parser.add_argument('--basis_emb_size', type=int, default=8,
                        help='Embedding size used in the basis transformation.')
    parser.add_argument('--num_blocks', type=int, default=6,
                        help='Number of building blocks.')
    parser.add_argument('--num_spherical', type=int, default=6,
                        help='Number of spherical harmonics.')
    parser.add_argument('--num_radial', type=int, default=6,
                        help='Number of radial basis functions.')
    parser.add_argument('--num_output_layers', type=int, default=2,
                        help='Number of linear layers for the output blocks.')
    parser.add_argument('--cutoff', type=float, default=5.0,
                        help='Cutoff distance for interatomic interactions.')
    parser.add_argument('--envelope_exponent', type=int, default=5,
                        help='Shape of the smooth cutoff.')
    parser.add_argument('--num_before_skip', type=int, default=1,
                        help='Number of residual layers in the interaction blocks before the skip connection.')
    parser.add_argument('--num_after_skip', type=int, default=2,
                        help='Number of residual layers in the interaction blocks after the skip connection.')
    parser.add_argument('--activation', type=str, default='SiLU',
                        choices=['SiLU', 'ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function.')

    # ffn parameters
    parser.add_argument('--ffn_inputs', nargs='+',
                        help='Name of columns to use as additional inputs for the FFN.')
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for FFN (defaults to out_channels).')
    parser.add_argument('--ffn_num_layers', type=int, default=3,
                        help='Number of layers in FFN.')
    parser.add_argument('--ffn_activation', type=str, default='LeakyReLU',
                        choices=['SiLU', 'ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function for the FFN.')
    parser.add_argument('--layer_norm', action='store_true', default=False,
                        help='Layer normalization in FFN.')
    parser.add_argument('--batch_norm', action='store_true', default=False,
                        help='Batch normalization in FFN.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability in FFN.')

    # inference parameters
    parser.add_argument('--model_params', type=str,
                        help='Path to yaml file containing model parameters. Used with fine-tuning or inference.')
    parser.add_argument('--state_dict', type=str,
                        help='Path to model checkpoint (.pt file). Used with fine-tuning or inference.')
    parser.add_argument('--sdtzer_path', type=str,
                        help='Path to standardizer used during training.')

    args = parser.parse_args(command_line_args)

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.out_channels

    return args
