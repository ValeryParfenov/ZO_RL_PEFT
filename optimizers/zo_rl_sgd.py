from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from .ldsd_diagnostics import alignment_diagnostics
from collections import defaultdict
import wandb

class ZO_RL_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = None,
            weight_decay: float = 0.0,
            tensor_sampling_type: str = "standard_normal",
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side",
            k: int = 10,
            variance: float = 1.0,
            lr_mu: Optional[float] = None,
            use_grad_first: bool = False,
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
        self.k = k
        self.variance = variance
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self.use_grad_first = use_grad_first
        
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0

                state['mu'] = torch.zeros_like(
                    param, 
                    memory_format=torch.preserve_format
                )
                # state['mu'] = torch.randn_like(
                #     param, 
                #     memory_format=torch.preserve_format
                # )
                # state['mu'] /= torch.linalg.norm(state['mu'])
                state['mu_old'] = state['mu'].detach().clone()
                state['mu_old_norm'] = torch.norm(state['mu_old']).item()**2

        
    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        loss_values = {}
        for _ in range(self.k):
            self.zo_random_seed = np.random.randint(1_000_000_000)
            self.generator.manual_seed(self.zo_random_seed)
            self._mu_pertrub(scaling_factor=1)
            if closure is not None:
                loss = closure()
                loss_values[self.zo_random_seed] = loss
            self.generator.manual_seed(self.zo_random_seed)
            self._mu_pertrub(scaling_factor=-1)
        
        optimal_seed = min(loss_values, key=loss_values.get)
        self.zo_random_seed = optimal_seed
        
        # self.zo_random_seed = np.random.randint(1_000_000_000)

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1)
        loss1 = closure()

        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=-2)
        loss2 = closure()
        
        self.projected_grad = (loss1 - loss2) / 2

        seeds = list(loss_values.keys())
        f_tensor = torch.tensor(list(loss_values.values()))
        f_sum = torch.sum(f_tensor)
        # Handle k=1 case to avoid division by zero
        if self.k > 1:
            coeff = (f_tensor * self.k - f_sum) / (self.k - 1)
        else:
            coeff = torch.zeros_like(f_tensor)  # When k=1, mu update term is zero


        # Restore parameters to original state (undo perturbations)
        self.generator.manual_seed(self.zo_random_seed)
        self._mu_pertrub(scaling_factor=1)

        # Alignment diagnostics at x_t with the policy mu_t that drew the candidates.
        # The generator is reseeded right below, so the trajectory does not change.
        if getattr(self, "diag_grads", None) is not None:
            self.last_diag = alignment_diagnostics(self, self.diag_grads, seeds, optimal_seed,
                                                   float(self.projected_grad))
            self.diag_grads = None

        self.generator.manual_seed(self.zo_random_seed)
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            eps = group['eps']
            momentum = group['momentum']
            
            for param in group['params']:
                state = self.state[param]
                state['step'] += 1
                device = param.device

                mu = state['mu']
                z = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(device)
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
        mu_norm_diff = None
        mu_grad_norm = None
        self.generator.manual_seed(self.zo_random_seed)
        for group in self.param_groups:
            eps = group['eps']
            lr = group['lr']
            for p in group['params']:    
                state = self.state[p]
                mu = state['mu']
                device = p.device
                mu_diff = torch.zeros_like(mu).to(device)
                for i, seed in enumerate(loss_values.keys()):
                    # Use the same z that was used during perturbation (saved in state['z'][seed])
                    z = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(device)
                    mu_diff += (mu - z) * coeff[i]
                    
                g_mu = -mu_diff / (self.k * (self.variance ** 2))
                state['mu'].add_(g_mu, alpha=-self.lr_mu)
                if mu_norm_diff is None:
                    mu_norm_diff = torch.linalg.norm(state['mu'] - state['mu_old'])**2 
                else:
                    mu_norm_diff += torch.linalg.norm(state['mu'] - state['mu_old'])**2 
                if mu_grad_norm is None:
                    mu_grad_norm = torch.linalg.norm(g_mu)**2 
                else:
                    mu_grad_norm += torch.linalg.norm(g_mu)**2 
        mu_norms = []
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                if 'mu' in state:
                    mu_norm = torch.norm(state['mu']).item()
                    mu_norms.append(mu_norm)
        
        # Get step from first parameter
        step = None
        try:
            first_param = list(self.param_groups[0]['params'])[0]
            step = self.state[first_param].get('step', None)
        except (KeyError, IndexError, AttributeError):
            pass
        
        if mu_norms:
            avg_mu_norm = sum(mu_norms) / len(mu_norms)
            if wandb.run is not None:
                wandb.log({"avg_mu_norm": avg_mu_norm})
            # Log to file if enabled
            try:
                from trainer import _optimizer_log_func
                if _optimizer_log_func is not None and step is not None:
                    _optimizer_log_func({"avg_mu_norm": avg_mu_norm}, step=step)
            except (ImportError, AttributeError):
                pass
        self.generator.manual_seed(self.zo_random_seed)
        
        avg_mu_norm_diff = torch.sqrt(mu_norm_diff) / len(mu_norms)
        avg_mu_grad_norm = torch.sqrt(mu_grad_norm) / len(mu_norms)
        # Convert tensors to Python numbers
        avg_mu_norm_diff_val = avg_mu_norm_diff.item() if torch.is_tensor(avg_mu_norm_diff) else float(avg_mu_norm_diff)
        avg_mu_grad_norm_val = avg_mu_grad_norm.item() if torch.is_tensor(avg_mu_grad_norm) else float(avg_mu_grad_norm)
        if wandb.run is not None:
            wandb.log({"avg_mu_norm_diff": avg_mu_norm_diff_val, "avg_mu_grad_norm": avg_mu_grad_norm_val})
        # Log to file if enabled
        try:
            from trainer import _optimizer_log_func
            if _optimizer_log_func is not None and step is not None:
                _optimizer_log_func({"avg_mu_norm_diff": avg_mu_norm_diff_val, "avg_mu_grad_norm": avg_mu_grad_norm_val}, step=step)
        except (ImportError, AttributeError):
            pass       

        return loss1
    
    def _mu_pertrub(self, scaling_factor: float = 1.0):
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:
                # Use the generator directly without resetting seed per parameter
                # This ensures all parameters use the same random sequence
                # self.generator.manual_seed(self.zo_random_seed)' 
                state = self.state[param]
                # if 'seed' not in state:
                #     state['seed'] = np.random.randint(1_000_000_000)
                # seed = state['seed'] 
                # self.generator.manual_seed(seed)
                mu = state['mu']    
                device = param.device
                z = torch.normal(mean=mu, std=self.variance, generator=self.generator).to(device)
                param.data.add_(z * eps * scaling_factor)
