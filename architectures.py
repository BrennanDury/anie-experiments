import torch
import torch.nn as nn
import math
from trim_transformer.transformer_layers import TrimTransformerEncoderLayer, TrimTransformerEncoder

def galerkin_init(param, gain=0.01, diagonal_weight=0.01):
    nn.init.xavier_uniform_(param, gain=gain)
    param.data += diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float, device=param.device))

class TrimTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, n_layers: int, scale: float | None = None, input_dim: int | None = None):
        super().__init__()
        norm_k = nn.LayerNorm(d_model//nhead)
        norm_v = nn.LayerNorm(d_model//nhead)
        encoder_layer = TrimTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_k=norm_k,
            norm_v=norm_v,
            q_weight_init=galerkin_init,
            k_weight_init=galerkin_init,
            v_weight_init=galerkin_init,
            scale=scale,
        )
        self.transformer_encoder = TrimTransformerEncoder(encoder_layer, num_layers=n_layers)
        if input_dim is not None:
            self.in_layer = nn.Linear(input_dim, d_model)
            self.out_layer = nn.Linear(d_model, input_dim)
        else:
            self.in_layer = None
            self.out_layer = None

    def forward(self, x: torch.Tensor, mask=None, use_kv_cache: bool = False, update_kv_cache: bool = False) -> torch.Tensor:
        if self.in_layer is not None:
            x = self.in_layer(x)
        x = self.transformer_encoder(x, mask=mask, use_kv_cache=use_kv_cache, update_kv_cache=update_kv_cache)
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x

    def clear_kv_cache(self):
        self.transformer_encoder.clear_kv_cache()

class PatchwiseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[32,32], K=[4,4], S=[4,4]):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(input_dim, hidden_dims[0],
                                     kernel_size=K,
                                     stride=S)

        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        B, T, H, W, Q = x.shape

        out = x.permute(0, 1, 4, 2, 3).reshape(B * T, Q, H, W)  # (B*T, Q, H, W)
        out = self.conv_layer1(out)  # (B*T, d_0, H', W')
        out = out.permute(0, 2, 3, 1)  # (B*T, H', W', d_0)

        out = self.fc1(out)  # (B*T, H', W', d_1)
        out = self.relu2(out)  # (B*T, H', W', d_1)
        out = self.fc2(out)  # (B*T, H', W', d_2)

        _BT, H_prime, W_prime, C_out = out.shape
        out = out.contiguous().view(B, T, H_prime, W_prime, C_out)  # (B, T, H', W', d_2)
        return out

class PatchwiseMLP_norm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[32,32],K=[4,4],S=[4,4]):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(input_dim, hidden_dims[0],
                                     kernel_size=K,
                                     stride=S)

        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], output_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dims[0])
        self.norm2 = nn.LayerNorm(hidden_dims[1])

    def forward(self, x):
        B, T, H, W, Q = x.shape

        out = x.permute(0, 1, 4, 2, 3).reshape(B * T, Q, H, W)  # (B*T, Q, H, W)
        out = self.conv_layer1(out)  # (B*T, hidden_dim, H', W')
        out = out.permute(0, 2, 3, 1)  # (B*T, H', W', hidden_dim)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.fc1(out)  # (B*T, H', W', hidden_ff)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.fc2(out)  # (B*T, H', W', out_dim)

        _BT, H_prime, W_prime, C_out = out.shape
        out = out.contiguous().view(B, T, H_prime, W_prime, C_out)  # (B, T, H', W', out_dim)
        return out

class TimestepwiseMLP(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_dims=[32,32], activation=nn.ELU):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.layer_sizes = hidden_dims
        total_layer_sizes = [in_shape.numel()] + hidden_dims + [out_shape.numel()]
        n_layers = len(total_layer_sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(total_layer_sizes[i],total_layer_sizes[i+1]) for i in range(n_layers)])
        self.activation = activation(inplace=True) 

    def forward(self, x):
        """
        x : B, T, Hp, Wp, C
        x : ... in_shape
        return : B, T, H, W, Q
        """
        x = x.flatten(-len(self.in_shape), -1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        x = x.reshape(*x.shape[:-1], *self.out_shape)
        return x

class DecoderWrapper(nn.Module):
    def __init__(self, decoder, Q, patch_shape):
        super(DecoderWrapper, self).__init__()
        self.decoder = decoder
        self.Q = Q
        self.ph, self.pw = patch_shape

    def forward(self, x):
        x = self.decoder(x)
        B, T, Hp, Wp, _ = x.shape
        x = x.view(B, T, Hp, Wp, self.ph, self.pw, self.Q)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.contiguous().view(B, T, Hp * self.ph, Wp * self.pw, self.Q)        
        return x

def make_positional_encoding(T, H, W) -> torch.Tensor:
    t_coord = torch.linspace(0, 1, T)  # (T,)
    row_coord = torch.linspace(0, 1, H)  # (H,)
    col_coord = torch.linspace(0, 1, W)  # (W,)

    time_enc = t_coord.view(T, 1, 1).expand(T, H, W)  # (T, H, W)
    row_enc  = row_coord.view(1, H, 1).expand(T, H, W)  # (T, H, W)
    col_enc  = col_coord.view(1, 1, W).expand(T, H, W)  # (T, H, W)

    out = torch.stack([time_enc, row_enc, col_enc], dim=-1)  # (T, H, W, P)
    return out

class PositionalEncoding(nn.Module):
    encoding: torch.Tensor
    def __init__(self, Tp, Hp, Wp):
        super().__init__()
        self.register_buffer('encoding', make_positional_encoding(Tp, Hp, Wp))

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """ 
        x : (B, Tp, Hp, Wp, C)
        t : int
        return : (B, Tp, Hp, Wp, C+P)
        """
        B, Tp = x.shape[0], x.shape[1]
        return torch.cat([self.encoding[t:t+Tp].unsqueeze(0).expand(B, -1, -1, -1, -1), x], dim=-1)

class PositionalUnencoding(nn.Module):
    def __init__(self, T, H, W):
        super().__init__()
        self.encoding = make_positional_encoding(T, H, W)

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        x : (B, Tp, Hp, Wp, C+P)
        return : (B, Tp, Hp, Wp, C)
        """
        return x[..., self.encoding.shape[-1]:]


def make_rotations(shape, base=10000):
    channel_dims, feature_dim = shape[:-1], shape[-1]
    k_max = feature_dim // (2 * len(channel_dims))

    assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'

    theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

    angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                        torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

    rotations = torch.polar(torch.ones_like(angles), angles)
    return rotations

class RoPENd(nn.Module):
    rotations: torch.Tensor
    def __init__(self, shape, base=10000):
        super(RoPENd, self).__init__()
        self.register_buffer('rotations', make_rotations(shape, base))

    def forward(self, x, t: int):
        B, Tp = x.shape[0], x.shape[1]
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = self.rotations[None, t:t+Tp] * x
        return torch.view_as_real(pe_x).flatten(-2)

class RoPENdUnencoding(nn.Module):
    rotations: torch.Tensor
    def __init__(self, shape, base=10000):
        super(RoPENdUnencoding, self).__init__()
        self.shape = shape
        self.register_buffer('rotations', 1/make_rotations(shape, base))

    def forward(self, x, t: int):
        B, Tp = x.shape[0], x.shape[1]
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = self.rotations[None, t:t+Tp] * x
        return torch.view_as_real(pe_x).flatten(-2)


class LearnedPositionalEncoding(nn.Module):
    encoding: torch.nn.Parameter

    def __init__(self, shape):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.empty(*shape))
        nn.init.zeros_(self.encoding)

    def forward(self, x, t: int):
        _B, Tp = x.shape[0], x.shape[1]
        return x + self.encoding[None, t : t + Tp]

class LearnedPositionalUnencoding(nn.Module):
    def __init__(self, learned_encoding_module: LearnedPositionalEncoding):
        super(LearnedPositionalUnencoding, self).__init__()
        self.learned_encoding_module = learned_encoding_module

    def forward(self, x, t: int):
        _B, Tp = x.shape[0], x.shape[1]
        return x - self.learned_encoding_module.encoding[None, t : t + Tp]