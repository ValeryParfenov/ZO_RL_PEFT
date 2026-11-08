"""Checks for the ZO-LDSD alignment diagnostics.

Run on a CUDA machine from the repository root:
    python -m tests.test_ldsd_diagnostics
"""
import logging
import math

import numpy as np
import torch

from optimizers import ZO_RL_SGD

logger = logging.getLogger(__name__)
DEVICE = "cuda"


def _problem(seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(8, 16, generator=gen).to(DEVICE))
    b = torch.nn.Parameter(torch.randn(16, generator=gen).to(DEVICE))

    def closure():
        return 0.5 * (w ** 2).sum() + 0.5 * ((b - 1.0) ** 2).sum()

    def grads():
        return [w.detach().clone(), (b - 1.0).detach().clone()]

    return [w, b], closure, grads


def _optimizer(params, lr_mu: float = 1e-2) -> ZO_RL_SGD:
    return ZO_RL_SGD(params, lr=1e-3, eps=1e-3, momentum=0.9, k=5, variance=1.0, lr_mu=lr_mu)


def test_summary_fields() -> None:
    params, closure, grads = _problem()
    opt = _optimizer(params)
    for _ in range(20):  # let mu move away from zero
        opt.step(closure)
    opt.diag_grads = grads()
    opt.step(closure)
    diag = opt.last_diag
    assert all(math.isfinite(c) and -1.0 <= c <= 1.0 for c in diag["cos_k"])
    assert diag["d"] == sum(p.numel() for p in params)
    assert abs(diag["C_mean"] - sum(c * c for c in diag["cos_k"]) / opt.k) < 1e-12
    assert abs(diag["C_iso"] - 1.0 / diag["d"]) < 1e-15
    assert diag["mu_norm"] > 0 and math.isfinite(diag["cos_mu"])
    assert abs(diag["mu_ratio"] - diag["mu_norm"] / math.sqrt(diag["d"])) < 1e-12
    assert abs(abs(diag["cos_est"]) - abs(diag["cos_sel"])) < 1e-15
    logger.info("cos_k=%s mu_ratio=%.3e cos_mu=%.3f", diag["cos_k"], diag["mu_ratio"], diag["cos_mu"])


def test_cosines_against_independent_draw() -> None:
    """Recompute cos(v_k, g) from scratch with the pre-step mu and the step's seeds."""
    params, closure, grads = _problem(1)
    opt = _optimizer(params)
    for _ in range(10):
        opt.step(closure)
    mu_before = [opt.state[p]["mu"].clone() for p in params]
    g = grads()
    np.random.seed(7)
    opt.diag_grads = g
    opt.step(closure)
    np.random.seed(7)
    seeds = [np.random.randint(1_000_000_000) for _ in range(opt.k)]
    gflat = torch.cat([x.flatten() for x in g]).double()
    for seed, cos in zip(seeds, opt.last_diag["cos_k"]):
        opt.generator.manual_seed(seed)
        v = torch.cat([torch.normal(mean=m, std=1.0, generator=opt.generator).flatten()
                       for m in mu_before]).double()
        expected = (v @ gflat / (v.norm() * gflat.norm())).item()
        assert abs(expected - cos) < 1e-9, (expected, cos)


def test_diagnostics_do_not_change_trajectory() -> None:
    params_a, closure_a, grads_a = _problem(2)
    params_b, closure_b, _ = _problem(2)
    opt_a, opt_b = _optimizer(params_a), _optimizer(params_b)
    for t in range(30):
        np.random.seed(t)
        opt_a.diag_grads = grads_a()
        opt_a.step(closure_a)
        np.random.seed(t)
        opt_b.step(closure_b)
    for pa, pb in zip(params_a, params_b):
        assert torch.equal(pa, pb), (pa - pb).abs().max()
    for pa, pb in zip(params_a, params_b):
        assert torch.equal(opt_a.state[pa]["mu"], opt_b.state[pb]["mu"])


def test_greedy_mu_zero_gives_isotropic_model() -> None:
    params, closure, grads = _problem(3)
    opt = _optimizer(params, lr_mu=0.0)
    opt.diag_grads = grads()
    opt.step(closure)
    diag = opt.last_diag
    assert diag["mu_norm"] == 0.0 and math.isnan(diag["cos_mu"])
    assert abs(diag["C_model"] - diag["C_iso"]) < 1e-15


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for test in (test_summary_fields, test_cosines_against_independent_draw,
                 test_diagnostics_do_not_change_trajectory, test_greedy_mu_zero_gives_isotropic_model):
        test()
        logger.info("%s passed", test.__name__)
