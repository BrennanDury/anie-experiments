import torch
import torch.nn as nn

class PicardIterations(nn.Module):
    """
    Regarding a model as an operator, this module performs Picard iterations of that operator.
    Precisely, at each step, the indices before -q are kept constant, and the indices from -q to -1 are updated
    by the rule r * Id + (1-r) * T[f], where T is the model.
    """
    def __init__(self, modules: nn.ModuleList, q: int, r: float = 0.5):
        """
        Args:
            modules (List[nn.Module]): The list of modules to apply Picard iterations to.
            q (int): The output dimension of the function.
            r (float): The relaxation parameter for the Picard step.
        """
        super().__init__()
        self.steps = nn.ModuleList([PicardStep(module, q, r) for module in modules])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for step in self.steps:
            x = step(x, *args, **kwargs)
        return x

    def clear_kv_cache(self):
        for step in self.steps:
            step.model.clear_kv_cache()

class PicardStep(nn.Module):
    """
    Regarding a model as an operator, this module performs a Picard iteration of that operator.
    Precisely, the indices before -q are kept constant, and the indices from -q to -1 are updated
    by the rule r * Id + (1-r) * T[f], where T is the model.
    """
    def __init__(self, model: nn.Module, q: int, r: float = 0.5):
        """
        Args:
            model (nn.Module): The model to apply the Picard step to.
            q (int): The number of previous steps to use for the Picard step.
            r (float): The relaxation parameter for the Picard step.
        """
        super().__init__()
        self.model = model
        self.q = q
        self.r = r

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        z = self.model(x, *args, **kwargs)
        y = self.r * x[..., -self.q:] + (1-self.r) * z[..., -self.q:]
        y_ = torch.cat([x[..., :-self.q], y], dim=-1)
        return y_

class ArbitraryIterations(nn.Module):
    """
    Apply a list of modules in sequence.
    """
    def __init__(self, modules: nn.ModuleList):
        """
        Args:
            modules (List[nn.Module]): The list of modules to apply.
        """
        super().__init__()
        self.steps = modules

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for step in self.steps:
            x = step(x, *args, **kwargs)
        return x

    def clear_kv_cache(self):
        for step in self.steps:
            step.clear_kv_cache()

def make_weight_shared_modules(make_module, n_modules: int) -> nn.ModuleList:
    module = make_module()
    return nn.ModuleList([module for _ in range(n_modules)])

def make_weight_unshared_modules(make_module, n_modules: int) -> nn.ModuleList:
    return nn.ModuleList([make_module() for _ in range(n_modules)])