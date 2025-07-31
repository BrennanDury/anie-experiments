import torch
import torch.nn as nn


class Iterations(nn.Module):
    def __init__(self, modules: nn.ModuleList, a: float = 1.0, b: float = 1.0, projection=None):
        super().__init__()
        self.steps = nn.ModuleList([Residual(module, a, b) for module in modules])

    def forward(self, x: torch.Tensor, projection=None, *args, **kwargs) -> torch.Tensor:
        if projection is None:
            projection = lambda x : x
        for step in self.steps:
            x = step(x, *args, **kwargs)
            x = projection(x)
        return x

    def clear_kv_cache(self):
        for step in self.steps:
            step.module.clear_kv_cache()

class Residual(nn.Module):
    def __init__(self, module: nn.Module, a=1.0, b=1.0):
        super().__init__()
        self.module = module
        self.a = a
        self.b = b  

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.a * x + self.b * self.module(x, *args, **kwargs)

def _tie_parameters(src: nn.Module, dst: nn.Module) -> None:
    """Assign every parameter in *dst* to reference the same object as the
    corresponding parameter in *src*.

    Both modules must have identical structure (``copy.deepcopy`` guarantees
    that).  Buffers are **not** shared, which means attributes like
    ``kv_cache`` that live inside sub-modules remain independent.
    """
    for name, param_src in src.named_parameters():
        # Traverse the attribute chain on ``dst`` to the leaf module that owns
        # the parameter so that we can overwrite its entry in _parameters.
        sub_dst = dst
        attr_chain = name.split('.')
        for attr in attr_chain[:-1]:
            sub_dst = getattr(sub_dst, attr)
        sub_dst._parameters[attr_chain[-1]] = param_src  # type: ignore[attr-defined]


def make_weight_shared_modules(make_module, n_modules: int) -> nn.ModuleList:
    """Create *n_modules* modules that share *parameters* but have independent
    buffers/state (e.g. separate ``kv_cache``) so that dropout masks and other
    per-call attributes are not reused across steps.
    """
    assert n_modules >= 1, "n_modules must be positive"

    # Build the master module once.
    master = make_module()
    modules = nn.ModuleList([master])

    # Create (n_modules - 1) clones that share parameters with *master*.
    import copy

    for _ in range(n_modules - 1):
        clone = copy.deepcopy(master)      # independent buffers & RNG state
        _tie_parameters(master, clone)     # share all parameters
        modules.append(clone)

    return modules

def make_weight_unshared_modules(make_module, n_modules: int) -> nn.ModuleList:
    return nn.ModuleList([make_module() for _ in range(n_modules)])