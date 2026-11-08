"""Alignment diagnostics for ZO-LDSD (reviewer question on the theoretical regime).

At selected steps the trainer computes the true minibatch gradient g = grad f(x_t) on the
same batch the ZO oracle sees and hands it to the optimizer. The optimizer then calls
`alignment_diagnostics` after the candidate losses are evaluated and the parameters are
restored, i.e. at x_t and with the policy mu_t that generated the candidates.

Logged quantities (d = number of trainable parameters, sigma = sampling std, `variance` arg):
  mu_ratio      ||mu_t|| / (sigma sqrt(d)); Lemma 3.6 / 3.7 need sigma = O(d^{-3/2} M), M <= ||mu_t||
  cos_mu        cos(mu_t, g), the angle phi_t of Theorem 3.5
  cos_k         cos(v_k, g) for each of the K sampled candidates v_k ~ N(mu_t, sigma^2 I)
  cos_sel       cos(v_sel, g) for the greedy argmin candidate used in the update
  cos_est       cos(g_x, g) for the ZO estimator g_x = projected_grad * v_sel
  C_mean        (1/K) sum_k cos_k^2, Monte Carlo estimate of E[C_t | F_{t-1}]
  C_model       (<mu_t, g_hat>^2 + sigma^2) / (||mu_t||^2 + sigma^2 d), large-d value of E[C_t | F_{t-1}]
  C_iso         1/d, the isotropic Gaussian baseline (Corollary 3.4)
Regenerating candidates reseeds the optimizer generator; the caller reseeds it afterwards,
so the training trajectory is unchanged.
"""
import math
from typing import Dict, List, Sequence

import torch

__all__ = ["alignment_diagnostics"]


def _candidate_stats(optimizer, params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor],
                     seed: int) -> tuple:
    """<v, g> and ||v||^2 for the candidate drawn with `seed` (same draw order as _mu_pertrub)."""
    optimizer.generator.manual_seed(seed)
    dot, sq = 0.0, 0.0
    for p, g in zip(params, grads):
        mu = optimizer.state[p]["mu"]
        z = torch.normal(mean=mu, std=optimizer.variance, generator=optimizer.generator).to(p.device)
        dot += torch.sum(z.double() * g.double()).item()
        sq += torch.sum(z.double() ** 2).item()
    return dot, sq


def alignment_diagnostics(optimizer, grads: Sequence[torch.Tensor], seeds: List[int],
                          selected_seed: int, projected_grad: float) -> Dict[str, float]:
    """Compute the quantities listed in the module docstring for one optimizer step."""
    params = [p for group in optimizer.param_groups for p in group["params"]]
    sigma = float(optimizer.variance)
    d = sum(p.numel() for p in params)
    g_sq = sum(torch.sum(g.double() ** 2).item() for g in grads)
    mu_sq = sum(torch.sum(optimizer.state[p]["mu"].double() ** 2).item() for p in params)
    mu_dot_g = sum(torch.sum(optimizer.state[p]["mu"].double() * g.double()).item()
                   for p, g in zip(params, grads))
    g_norm, mu_norm = math.sqrt(g_sq), math.sqrt(mu_sq)

    cos_k = []
    for seed in seeds:
        dot, sq = _candidate_stats(optimizer, params, grads, seed)
        cos_k.append(dot / (math.sqrt(sq) * g_norm) if sq > 0 and g_norm > 0 else float("nan"))
    cos_sel = cos_k[seeds.index(selected_seed)]
    cos_mu = mu_dot_g / (mu_norm * g_norm) if mu_norm > 0 and g_norm > 0 else float("nan")
    mu_dot_ghat = mu_dot_g / g_norm if g_norm > 0 else 0.0
    sign = 1.0 if projected_grad >= 0 else -1.0
    return {
        "d": d,
        "sigma": sigma,
        "grad_norm": g_norm,
        "mu_norm": mu_norm,
        "mu_ratio": mu_norm / (sigma * math.sqrt(d)),
        "cos_mu": cos_mu,
        "cos_k": cos_k,
        "cos_sel": cos_sel,
        "cos_est": sign * cos_sel,
        "C_mean": sum(c * c for c in cos_k) / len(cos_k),
        "C_sel": cos_sel * cos_sel,
        "C_model": (mu_dot_ghat ** 2 + sigma ** 2) / (mu_sq + sigma ** 2 * d),
        "C_iso": 1.0 / d,
    }
