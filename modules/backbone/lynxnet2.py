import torch.nn as nn

from modules.commons.common_layers import SwiGLU, ATanGLU, Transpose


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0., glu_type='swiglu'):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if glu_type == 'swiglu':
            _glu = SwiGLU()
        elif glu_type == 'atanglu':
            _glu = ATanGLU()
        else:
            raise ValueError(f'{glu_type} is not a valid activation')
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            _glu,
            nn.Linear(inner_dim, inner_dim * 2),
            _glu,
            nn.Linear(inner_dim, dim),
            _dropout
        )

    def forward(self, x):
        return x + self.net(x)


class LYNXNet2(nn.Module):
    def __init__(self, in_dims, out_dims, *, num_layers=6, num_channels=512, expansion_factor=1, kernel_size=31,
                 dropout_rate=0.0, glu_type='swiglu'):
        """
        LYNXNet2(Linear Gated Depthwise Separable Convolution Network Version 2)
        """
        super().__init__()
        self.in_dims = in_dims
        self.input_projection = nn.Linear(in_dims, num_channels)
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    dropout=dropout_rate,
                    glu_type=glu_type
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, out_dims)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x):
        """
        :param x: [B, T, C_in]
        :return: [B, T, C_out]
        """
        x = self.input_projection(x)

        for layer in self.residual_layers:
            x = layer(x)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x)

        return x
