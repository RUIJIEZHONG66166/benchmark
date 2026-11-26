import torch

# Recent PyTorch removed the public torch.jit.interface (pytorch/pytorch#203621).
# TensorToTensor is only a type annotation in eager mode, so fall back to the
# internal implementation, or to a no-op if that is gone as well.
_jit_interface = getattr(torch.jit, "interface", None)
if _jit_interface is None:
    try:
        from torch.jit._script import interface as _jit_interface
    except ImportError:

        def _jit_interface(obj):
            return obj


@_jit_interface
class TensorToTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
