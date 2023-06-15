from typing import Tuple, Optional, List, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, reduce, repeat


@torch.jit.script
def sinkhorn(
    C: Tensor,
    a: Tensor,
    b: Tensor,
    n_sh_iters: int = 5,
    temperature: float = 1,
    u: Union[Tensor, None] = None,
    v: Union[Tensor, None] = None,
    min_clamp: float = 1e-30,
) -> Tuple[Tensor, Tensor, Tensor]:
    p = -C / temperature
    
    # NOTE: clamp to avoid -inf;
    # exact value decides minimal attention per location/slot
    log_a = torch.log(a.clamp(min=min_clamp))
    log_b = torch.log(b.clamp(min=min_clamp))

    if u is None:
        u = torch.zeros_like(a)
    if v is None:
        v = torch.zeros_like(b)

    for _ in range(n_sh_iters):
        u = log_a - torch.logsumexp(p + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(p + u.unsqueeze(2), dim=1)

    logT = p + u.unsqueeze(2) + v.unsqueeze(1)
    return logT.exp(), u, v


@torch.enable_grad()
def minimize_entropy_of_sinkhorn(
    C_0, a, b, noise=None, mesh_lr=1, n_mesh_iters=4, n_sh_iters=5, reuse_u_v=True
):
    if noise is None:
        noise = torch.randn_like(C_0)

    C_t = C_0 + 0.001 * noise
    C_t.requires_grad_(True)

    u = None
    v = None
    for i in range(n_mesh_iters):
        attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, n_sh_iters=n_sh_iters)

        if not reuse_u_v:
            u = v = None

        entropy = reduce(
            torch.special.entr(attn.clamp(min=1e-20, max=1)), "n a b -> n", "mean"
        ).sum()
        (grad,) = torch.autograd.grad(entropy, C_t, retain_graph=True)
        grad = F.normalize(grad + 1e-20, dim=[1, 2])
        C_t = C_t - mesh_lr * grad

    # attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, num_sink=num_sink_iters)

    if not reuse_u_v:
        u = v = None

    return C_t, u, v
