import torch
import torch.nn as nn

class AcausalWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        """
        model : (B, (T+1)*Hp*Wp, C) -> (B, (T+1)*Hp*Wp, C)
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def clear_kv_cache(self):
        self.model.clear_kv_cache()

class GenerateWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        """
        model : (B, 1*Hp*Wp, C) -> (B, 1*Hp*Wp, C)
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, use_kv_cache=True, update_kv_cache=True)

    def clear_kv_cache(self):
        self.model.clear_kv_cache()

class OneStepWrapper(nn.Module):
    def __init__(self, model: nn.Module, mask: torch.Tensor):
        """
        model : (B, T*Hp*Wp, C) -> (B, T*Hp*Wp, C)
        """
        super().__init__()
        self.model = model
        self.mask = mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, mask=self.mask)

    def clear_kv_cache(self):
        self.model.clear_kv_cache()