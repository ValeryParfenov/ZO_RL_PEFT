"""LOREN: low-rank curvature ZO optimizer with RLOO variance reduction.

Seung et al., "Low-Rank Curvature for Zeroth-Order Optimization in LLM
Fine-Tuning" (AAAI 2026). Ported from the official implementation
(github.com/hseung88/loren, model/glue_transformer_loren.py) onto this
repository's optimizer interface. The update logic follows the original
line by line; only the random source (self.generator instead of the global
torch seed) and the place of the learning rate (param_groups) differ.

Per iteration: K one-sided queries f(x + eps * Sigma^{1/2} u_k), where
Sigma^{-1} = damping * I + a a^T per block, with the rank-1 vector a shared
across the rows of each weight matrix. Parameters and a are updated with the
RLOO weights f_k - mean(f).

One deliberate deviation: for 1-D parameters the original reduces the
gradient of a with `.view(shape).sum(0)`, which collapses it to a scalar that
is then broadcast over a. Here a 1-D parameter gets the elementwise gradient
(mfactor = 1, so a has the parameter's own size).
"""
import math
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch

from .base import ZeroOrderOptimizer


class LOREN(ZeroOrderOptimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        num_queries: int = 6,
        damping: float = 0.1,
        lr_cov: float = 1e-3,
        tensor_sampling_type: str = "standard_normal",
        matrix_sampling_type: Optional[str] = None,
        perturbation_mode: str = "one_side",
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )
        if num_queries < 2:
            raise ValueError("LOREN needs at least 2 queries for the RLOO estimator")
        self.num_queries = num_queries
        self.damping = damping
        self.d_sqrt = math.sqrt(damping)
        self.lr_cov = lr_cov

        init_gen = torch.Generator(device=self.generator.device)
        init_gen.manual_seed(np.random.randint(1_000_000_000))
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                evec_size = p.size(1) if p.ndim >= 2 else p.numel()
                state['evec_a'] = torch.randn(evec_size, device=p.device, dtype=p.dtype, generator=init_gen)
                state['mfactor'] = p.numel() // evec_size
                if p.ndim >= 2:
                    state['momentum_buffer'] = torch.zeros_like(p)

    def _sqrt_cov_u(self, p: torch.Tensor):
        """Return (Sigma^{1/2} u, u, a_flat, dot_au, sq_norm_a, damping_sq_norm_a) for a fresh u."""
        state = self.state[p]
        a = state['evec_a']
        a_flat = a.view(1, -1).expand(state['mfactor'], -1).reshape(-1)
        u_vec = torch.normal(mean=0, std=1, size=p.shape, device=p.device,
                             generator=self.generator).view(-1)
        sq_norm_a = torch.dot(a_flat, a_flat)
        dot_au = torch.dot(a_flat, u_vec)
        # kept as a tensor: .item() here forces a GPU sync per parameter and query
        damping_sq_norm_a = torch.sqrt(self.damping + sq_norm_a)
        alpha = (self.d_sqrt + damping_sq_norm_a) * dot_au / (sq_norm_a * damping_sq_norm_a)
        sqrt_cov_u = u_vec.sub(alpha * a_flat).view(p.shape)
        return sqrt_cov_u, u_vec, a_flat, dot_au, sq_norm_a, damping_sq_norm_a

    def _perturb(self, seed: int, scaling_factor: float) -> None:
        self.generator.manual_seed(seed)
        for group in self.param_groups:
            eps = group['eps']
            for p in group['params']:
                sqrt_cov_u = self._sqrt_cov_u(p)[0]
                p.data.add_(sqrt_cov_u, alpha=scaling_factor * eps)

    @torch.no_grad()
    def step(self, closure=None):
        seeds = [np.random.randint(1_000_000_000) for _ in range(self.num_queries)]
        losses = []
        for seed in seeds:
            self._perturb(seed, scaling_factor=1.0)
            losses.append(closure())
            self._perturb(seed, scaling_factor=-1.0)

        fvals = torch.stack([l.float() for l in losses]).cpu()
        fweight = fvals - fvals.mean()
        k_inv = 1.0 / (self.num_queries - 1)

        for k, seed in enumerate(seeds):
            self.generator.manual_seed(seed)
            for group in self.param_groups:
                lr, eps = group['lr'], group['eps']
                wd, beta1 = group['weight_decay'], group['momentum']
                eps_inv = 1.0 / (eps * (self.num_queries - 1))
                for p in group['params']:
                    state = self.state[p]
                    sqrt_cov_u, u_vec, a_flat, dot_au, sq_norm_a, dsn = self._sqrt_cov_u(p)
                    w = fweight[k].item()

                    gx = sqrt_cov_u.mul(w * eps_inv)
                    if p.ndim >= 2:
                        state['momentum_buffer'].mul_(beta1).add_(gx)
                        gx = state['momentum_buffer']
                    p.data.mul_(1 - lr * wd).add_(gx, alpha=-lr)

                    c1 = dot_au * u_vec - a_flat
                    c2 = (self.d_sqrt + dsn) * (dot_au ** 2 - sq_norm_a) * a_flat / (sq_norm_a * dsn)
                    a_grad = (w * eps ** 2) * (c1 - c2) / dsn
                    a_grad = a_grad.view(p.shape).sum(0) * k_inv if p.ndim >= 2 else a_grad * k_inv
                    state['evec_a'].mul_(1 - self.lr_cov * wd).add_(a_grad, alpha=-self.lr_cov)
                    if k == 0:
                        state['step'] += 1

        return losses[0]
