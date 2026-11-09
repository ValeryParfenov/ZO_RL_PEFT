"""Smoke tests for ZoAR, LOREN and greedy ZO-SGD (zo_rl_sgd with lr_mu=0).

Run on a CUDA machine from the repository root:
    python -m tests.test_new_optimizers
"""
import logging

import torch

from optimizers import LOREN, ZO_RL_SGD, ZoAR

logger = logging.getLogger(__name__)
DEVICE = "cuda"


def _quadratic_problem(seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(8, 16, generator=gen).to(DEVICE))
    b = torch.nn.Parameter(torch.randn(16, generator=gen).to(DEVICE))
    target_w = torch.zeros_like(w)

    def closure():
        return 0.5 * ((w - target_w) ** 2).sum() + 0.5 * (b ** 2).sum()

    return [w, b], closure


def test_zoar_running_sums_match_exact() -> None:
    params, closure = _quadratic_problem()
    opt = ZoAR(params, lr=1e-2, eps=1e-2, num_queries=2, history=3)
    for _ in range(12):
        opt.step(closure)
        # exact sums over the buffer
        exact = {id(p): (torch.zeros_like(p), torch.zeros_like(p)) for p in params}
        for seed, y in opt.buffer:
            opt.generator.manual_seed(seed)
            for p in params:
                u = opt._sample(p)
                exact[id(p)][0].add_(u, alpha=y)
                exact[id(p)][1].add_(u)
        assert len(opt.buffer) <= 6
        for p in params:
            st = opt.state[p]
            torch.testing.assert_close(st['sum_yu'], exact[id(p)][0], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(st['sum_u'], exact[id(p)][1], rtol=1e-4, atol=1e-4)


def _decreases(opt_factory, steps: int) -> tuple:
    params, closure = _quadratic_problem()
    before = closure().item()
    opt = opt_factory(params)
    for _ in range(steps):
        opt.step(closure)
    return before, closure().item()


def test_zoar_decreases_loss() -> None:
    before, after = _decreases(lambda ps: ZoAR(ps, lr=5e-3, eps=1e-2, num_queries=2, history=5), 2000)
    assert after < 0.5 * before, (before, after)


def test_zoar_step_baseline_survives_minibatch_noise() -> None:
    """A per-step loss offset (a new minibatch every step) leaves the step baseline untouched."""
    results = {}
    for mode in ("history", "step"):
        params, base = _quadratic_problem()
        gen = torch.Generator().manual_seed(1)
        offset = [0.0]

        def closure():
            return base() + offset[0]

        before = base().item()
        opt = ZoAR(params, lr=5e-3, eps=1e-2, num_queries=2, history=5, baseline=mode)
        for _ in range(2000):
            offset[0] = 5.0 * torch.randn(1, generator=gen).item()
            opt.step(closure)
        results[mode] = base().item() / before
    logger.info("relative loss after 2000 noisy steps: %s", results)
    assert results["step"] < 0.5, results


def test_loren_decreases_loss_and_restores_params() -> None:
    params, closure = _quadratic_problem()
    opt = LOREN(params, lr=1e-3, eps=1e-3, momentum=0.9, num_queries=6, damping=0.1, lr_cov=1e-3)
    snapshot = [p.detach().clone() for p in params]
    opt._perturb(123, 1.0)
    opt._perturb(123, -1.0)
    for p, s in zip(params, snapshot):
        torch.testing.assert_close(p.detach(), s, rtol=0, atol=1e-6)
    before, after = _decreases(
        lambda ps: LOREN(ps, lr=1e-3, eps=1e-3, momentum=0.9, num_queries=6, damping=0.1, lr_cov=1e-3), 400)
    assert after < 0.5 * before, (before, after)


def test_greedy_sgd_keeps_mu_zero() -> None:
    params, closure = _quadratic_problem()
    opt = ZO_RL_SGD(params, lr=1e-3, eps=1e-3, momentum=0.9, k=5, variance=1.0, lr_mu=0.0)
    for _ in range(5):
        opt.step(closure)
    for p in params:
        assert torch.count_nonzero(opt.state[p]['mu']) == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            logger.info("PASS %s", name)
