import torch
import torch.nn as nn
from trim_transformer.transformer_layers import TrimTransformerEncoderLayer, TrimTransformerEncoder

def galerkin_init(param, gain=0.01, diagonal_weight=0.01):
    nn.init.xavier_uniform_(param, gain=gain)
    param.data += diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float, device=param.device))

class TrimTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, n_layers: int, scale: float | None = None):
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

    def forward(self, x: torch.Tensor, mask=None, use_kv_cache: bool = False, update_kv_cache: bool = False) -> torch.Tensor:
        return self.transformer_encoder(x, mask=mask, use_kv_cache=use_kv_cache, update_kv_cache=update_kv_cache)

    def clear_kv_cache(self):
        self.transformer_encoder.clear_kv_cache()

class PatchwiseMLP(nn.Module):
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=64,K=[4,4],S=[4,4]):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(dim, hidden_dim,
                                     kernel_size=K,
                                     stride=S)

        self.fc1 = nn.Linear(hidden_dim, hidden_ff)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)

    def forward(self, x):
        B, T, H, W, Q = x.shape

        out = x.permute(0, 1, 4, 2, 3).reshape(B * T, Q, H, W)  # (B*T, Q, H, W)
        out = self.conv_layer1(out)  # (B*T, hidden_dim, H', W')
        out = out.permute(0, 2, 3, 1)  # (B*T, H', W', hidden_dim)

        out = self.fc1(out)  # (B*T, H', W', hidden_ff)
        out = self.relu2(out)  # (B*T, H', W', hidden_ff)
        out = self.fc2(out)  # (B*T, H', W', out_dim)

        _BT, H_prime, W_prime, C_out = out.shape
        out = out.contiguous().view(B, T, H_prime, W_prime, C_out)  # (B, T, H', W', out_dim)
        return out

class TimestepwiseMLP(nn.Module):
    def __init__(self, in_shape, layer_sizes, out_shape, activation=nn.ELU):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.layer_sizes = layer_sizes
        total_layer_sizes = [in_shape.numel()] + layer_sizes + [out_shape.numel()]
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
        return torch.cat([x, self.encoding[t:t+Tp].unsqueeze(0).expand(B, -1, -1, -1, -1)], dim=-1)

class PositionalUnencoding(nn.Module):
    def __init__(self, T, H, W):
        super().__init__()
        self.encoding = make_positional_encoding(T, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, Tp, Hp, Wp, C+P)
        return : (B, Tp, Hp, Wp, C)
        """
        return x[..., :-self.encoding.shape[-1]]