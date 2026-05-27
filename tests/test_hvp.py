import torch

from solutions._01_krylov import (
    flat_params,
    hvp_finite_difference,
    hvp_double_backward,
    hvp_jvp_of_grad,
    reference_hvp,
)
from src.tiny_models import toy_mlp


def _setup():
    torch.manual_seed(0)
    model = toy_mlp(seed=1)
    X = torch.randn(8, 20)
    y = torch.randint(0, 4, (8,))
    P = sum(p.numel() for p in model.parameters())
    v = torch.randn(P)
    return model, X, y, v


def test_all_three_hvp_agree():
    model, X, y, v = _setup()
    h1 = hvp_double_backward(model, X, y, v)
    h2 = hvp_jvp_of_grad(model, X, y, v)
    h3 = hvp_finite_difference(model, X, y, v, eps=1e-3)
    assert torch.allclose(h1, h2, atol=1e-5)
    assert torch.allclose(h1, h3, atol=5e-3)


def test_reference_hvp_matches_explicit_hessian():
    from torch.func import functional_call
    model, X, y, v = _setup()
    h_ref = reference_hvp(model, X, y, v)

    params_named = {n: p for n, p in model.named_parameters()}

    def loss_fn(theta_flat):
        pd, i = {}, 0
        for n, p in params_named.items():
            pd[n] = theta_flat[i : i + p.numel()].view_as(p)
            i += p.numel()
        return torch.nn.functional.cross_entropy(functional_call(model, pd, (X,)), y)

    theta = flat_params(model)
    H = torch.autograd.functional.hessian(loss_fn, theta)
    h_explicit = H @ v
    assert torch.allclose(h_ref, h_explicit, atol=1e-4)


def test_flat_params_roundtrip():
    model = toy_mlp(seed=2)
    flat = flat_params(model)
    P = sum(p.numel() for p in model.parameters())
    assert flat.shape == (P,)
