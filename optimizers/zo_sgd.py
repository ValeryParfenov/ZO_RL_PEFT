from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict

class ZO_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = None,
            weight_decay: float = 0.0,
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
            k: int = 1,
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
        # k > 1: average k two-sided Gaussian estimates per step (2k forwards), as in the
        # authors' dev branch (llm_ft/optimizers/zo_sgd.py); k = 1 is the original update.
        self.k = max(1, k)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        if self.k > 1:
            return self._step_averaged(closure)
        loss1, loss2 = None, None

        self.zo_random_seed = np.random.randint(1_000_000_000)

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1)
        loss1 = closure()

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=-2)
        loss2 = closure()
        
        self.projected_grad = (loss1 - loss2) / 2

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1)       

        self.generator.manual_seed(self.zo_random_seed)
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            eps = group['eps']
            momentum = group['momentum']
            
            for param in group['params']:
                state = self.state[param]
                state['step'] += 1

                z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                grad = (z * self.projected_grad) / eps
                if momentum is not None and momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad    
                param.data.add_(update, alpha=-lr)

        return loss1

    @torch.no_grad()
    def _step_averaged(self, closure):
        """Mean of k two-sided estimates along independent Gaussian directions."""
        seeds, grads_proj, losses = [], [], []
        for _ in range(self.k):
            seed = np.random.randint(1_000_000_000)
            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=1)
            loss1 = closure()
            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=-2)
            loss2 = closure()
            self.generator.manual_seed(seed)
            self._mu_pertrub(scaling_factor=1)
            seeds.append(seed)
            grads_proj.append(float((loss1 - loss2) / 2))
            losses.append(loss1)
        self.zo_random_seed = seeds[0]
        self.projected_grad = sum(grads_proj) / self.k

        grads = {}
        for seed, pg in zip(seeds, grads_proj):
            self.generator.manual_seed(seed)
            for group in self.param_groups:
                for param in group['params']:
                    z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                    g = z * (pg / (group['eps'] * self.k))
                    grads[param] = g if param not in grads else grads[param].add_(g)

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for param in group['params']:
                state = self.state[param]
                state['step'] += 1
                grad = grads[param]
                if momentum is not None and momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad
                param.data.add_(update, alpha=-lr)
        return sum(losses) / self.k
    
    def _mu_pertrub(self, scaling_factor: float = 1.0):
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:
                state = self.state[param]
                z = torch.normal(mean=0, std=1, size=param.shape, device=param.device, generator=self.generator)
                param.data.add_(z * eps * scaling_factor)

