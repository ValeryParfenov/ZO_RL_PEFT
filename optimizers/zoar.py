"""ZoAR: ZO optimization with Averaged baseline and query Reuse.

Qiu et al., "Zeroth-Order Optimization is Secretly Single-Step Policy
Optimization" (arXiv:2506.14460), Algorithm 1, with the R-AdaZO update rule
used in their LLM fine-tuning experiments.

Per iteration: K one-sided queries y_k = f(x + mu * u_k), u_k ~ N(0, I).
The history H keeps the last N*K pairs (u, y) as random seeds. Then
    b = mean_{(u,y) in H} y
    g = 1 / (mu * (|H| - 1)) * sum_{(u,y) in H} (y - b) u
    m = beta1 m + (1 - beta1) g
    v = beta2 v + (1 - beta2) m^2
    x = x - lr * m / sqrt(v + adam_eps)

The sums S_yu = sum y u and S_u = sum u are kept as running buffers, so a
step regenerates only the K new and the K evicted directions instead of all
N*K, and g = (S_yu - b S_u) / (mu (|H| - 1)).

baseline="step" is our variant for minibatch training: in fine-tuning every step
sees a new minibatch, and the loss differs between minibatches far more than
between two directions on one minibatch, so y - b with b averaged over N steps is
mostly minibatch noise. Here each step's K losses are centered by their own mean,
y~_k = y_k - mean_j y_j (same minibatch), the centered values are kept in the history,
and g = sum_H y~ u / (mu |H| (1 - 1/K)); E[(y_k - mean y) u_k] = (1 - 1/K) E[y_k u_k].
"""
from collections import deque
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch

from .base import ZeroOrderOptimizer


class ZoAR(ZeroOrderOptimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        num_queries: int = 2,
        history: int = 15,
        adam_eps: float = 1e-8,
        tensor_sampling_type: str = "standard_normal",
        matrix_sampling_type: Optional[str] = None,
        perturbation_mode: str = "one_side",
        baseline: str = "history",
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )
        if num_queries * history < 2:
            raise ValueError("ZoAR needs at least 2 samples in the history (K * N >= 2)")
        if baseline not in ("history", "step") or (baseline == "step" and num_queries < 2):
            raise ValueError("baseline must be 'history' or 'step' (the latter needs K >= 2)")
        self.baseline = baseline
        self.num_queries = num_queries
        self.history = history
        self.adam_eps = adam_eps
        self.buffer: deque = deque()  # (seed, loss) pairs, oldest first

        for group in self.param_groups:
            group['betas'] = betas
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['sum_yu'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['sum_u'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def _sample(self, p: torch.Tensor) -> torch.Tensor:
        return torch.normal(mean=0, std=1, size=p.shape, device=p.device, generator=self.generator)

    def _perturb(self, seed: int, scaling_factor: float) -> None:
        self.generator.manual_seed(seed)
        for group in self.param_groups:
            eps = group['eps']
            for p in group['params']:
                p.data.add_(self._sample(p), alpha=eps * scaling_factor)

    def _accumulate(self, seed: int, loss: float, sign: float) -> None:
        """Add (sign=+1) or remove (sign=-1) one sample from the running sums."""
        self.generator.manual_seed(seed)
        for group in self.param_groups:
            for p in group['params']:
                u = self._sample(p)
                state = self.state[p]
                state['sum_yu'].add_(u, alpha=sign * loss)
                state['sum_u'].add_(u, alpha=sign)

    @torch.no_grad()
    def step(self, closure=None):
        first_loss = None
        new_samples = []
        for _ in range(self.num_queries):
            seed = np.random.randint(1_000_000_000)
            self._perturb(seed, scaling_factor=1.0)
            loss = closure()
            self._perturb(seed, scaling_factor=-1.0)
            if first_loss is None:
                first_loss = loss
            new_samples.append((seed, loss.item()))

        if self.baseline == "step":
            mean_y = sum(y for _, y in new_samples) / len(new_samples)
            new_samples = [(seed, y - mean_y) for seed, y in new_samples]
        for seed, y in new_samples:
            self.buffer.append((seed, y))
            self._accumulate(seed, y, sign=1.0)
        while len(self.buffer) > self.num_queries * self.history:
            seed, y = self.buffer.popleft()
            self._accumulate(seed, y, sign=-1.0)

        n = len(self.buffer)
        if self.baseline == "step":
            baseline, denom_n = 0.0, n * (1.0 - 1.0 / self.num_queries)
        else:
            baseline, denom_n = sum(y for _, y in self.buffer) / n, n - 1
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            scale = 1.0 / (group['eps'] * denom_n)
            for p in group['params']:
                state = self.state[p]
                state['step'] += 1
                grad = (state['sum_yu'] - baseline * state['sum_u']).mul_(scale)
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(
                    state['exp_avg'], state['exp_avg'], value=1.0 - beta2)
                denom = state['exp_avg_sq'].add(self.adam_eps).sqrt_()
                p.data.addcdiv_(state['exp_avg'], denom, value=-lr)

        return first_loss
