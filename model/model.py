import torch
import torch.nn as nn

from model.dimenet_pp import DimeNetPlusPlus
from model.mlp import MLP


class ReactionModel(nn.Module):
    def __init__(self,
                 # dimenet_pp
                 hidden_channels=50,
                 out_emb_channels=50,
                 out_channels=50,
                 int_emb_size=64,
                 basis_emb_size=8,
                 num_blocks=6,
                 num_spherical=6,
                 num_radial=6,
                 num_output_layers=2,
                 cutoff=5.0,
                 envelope_exponent=5,
                 num_before_skip=1,
                 num_after_skip=2,
                 activation=nn.SiLU(),
                 # MLP
                 ffn_hidden_size=50,
                 out_dim=1,
                 ffn_num_layers=3,
                 ffn_activation=nn.LeakyReLU(),
                 dropout=0.0,
                 layer_norm=False,
                 batch_norm=False,
                 num_additional_ffn_inputs=0,
                 ):
        super(ReactionModel, self).__init__()

        self.dimenet_pp = DimeNetPlusPlus(hidden_channels=hidden_channels,
                                          out_channels=out_channels,
                                          num_blocks=num_blocks,
                                          int_emb_size=int_emb_size,
                                          basis_emb_size=basis_emb_size,
                                          out_emb_channels=out_emb_channels,
                                          num_spherical=num_spherical,
                                          num_radial=num_radial,
                                          num_output_layers=num_output_layers,
                                          cutoff=cutoff,
                                          envelope_exponent=envelope_exponent,
                                          num_before_skip=num_before_skip,
                                          num_after_skip=num_after_skip,
                                          act=activation,
                                          )
        out_channels += num_additional_ffn_inputs
        self.ffn = MLP(in_dim=out_channels,
                       h_dim=ffn_hidden_size,
                       out_dim=out_dim,
                       num_layers=ffn_num_layers,
                       activation=ffn_activation,
                       dropout=dropout,
                       layer_norm=layer_norm,
                       batch_norm=batch_norm,
                       )

    def forward(self, ts_z, ts_coords, r_z, r_coords, r_z_batch, additional_ffn_inputs=None):
        diff = self.dimenet_pp(ts_z, ts_coords, r_z_batch) - self.dimenet_pp(r_z, r_coords, r_z_batch)
        if additional_ffn_inputs is not None:
            diff = torch.cat((diff, additional_ffn_inputs.float()), dim=-1)
        out = self.ffn(diff)

        return out
