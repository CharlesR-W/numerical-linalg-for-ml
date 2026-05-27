"""Reference implementations for Notebook 1 (Krylov methods).

These are the source of truth that the notebook's exercises check against.
Later notebooks also import from here for ground-truth HVPs.
"""
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    """Return a single 1D tensor concatenating all parameters."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _params_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach() for n, p in model.named_parameters()}


def _unflatten(flat: torch.Tensor, ref: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out, i = {}, 0
    for n, p in ref.items():
        out[n] = flat[i : i + p.numel()].view_as(p)
        i += p.numel()
    return out


def _loss(params_dict, model, X, y):
    logits = functional_call(model, params_dict, (X,))
    return F.cross_entropy(logits, y)


def hvp_double_backward(model, X, y, v):
    """HVP via double backward.  Conceptually simplest, fp behaves well."""
    params = {n: p for n, p in model.named_parameters()}
    flat = torch.cat([p.reshape(-1) for p in params.values()]).requires_grad_(True)
    pd = _unflatten(flat, params)
    loss = _loss(pd, model, X, y)
    g_flat = torch.autograd.grad(loss, flat, create_graph=True)[0]
    hv = torch.autograd.grad(g_flat @ v, flat, retain_graph=False)[0]
    return hv.detach()


def hvp_jvp_of_grad(model, X, y, v):
    """HVP via JVP of grad — single forward+backward via torch.func."""
    params = _params_dict(model)

    def flat_grad(flat):
        pd = _unflatten(flat, params)
        g_dict = grad(_loss)(pd, model, X, y)
        return torch.cat([g_dict[n].reshape(-1) for n in params])

    flat = flat_params(model)
    _, hvp_flat = jvp(flat_grad, (flat,), (v,))
    return hvp_flat


def hvp_finite_difference(model, X, y, v, eps: float = 1e-3):
    """HVP by central differences on the gradient.  Slow but reference-of-references."""
    params = list(model.parameters())

    def grad_at(direction_flat):
        with torch.no_grad():
            i = 0
            for p in params:
                p.add_(direction_flat[i : i + p.numel()].view_as(p))
                i += p.numel()
        loss = F.cross_entropy(model(X), y)
        g = torch.autograd.grad(loss, list(model.parameters()))
        g_flat = torch.cat([gi.reshape(-1) for gi in g])
        with torch.no_grad():
            i = 0
            for p in params:
                p.sub_(direction_flat[i : i + p.numel()].view_as(p))
                i += p.numel()
        return g_flat

    g_plus = grad_at(eps * v)
    g_minus = grad_at(-eps * v)
    return (g_plus - g_minus) / (2 * eps)


reference_hvp = hvp_double_backward


def power_iteration(matvec, dim, num_iters=200, tol=1e-10, seed=0):
    """Plain power iteration.

    Returns: (top eigenvalue estimate, top eigvec, list of per-iter Rayleigh
    quotient errors vs final estimate).
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    v = v / v.norm()
    history = []
    eigval = 0.0
    for _ in range(num_iters):
        Av = matvec(v)
        new_eigval = (v @ Av).item()
        v = Av / (Av.norm() + 1e-30)
        history.append(abs(new_eigval - eigval))
        if abs(new_eigval - eigval) < tol:
            eigval = new_eigval
            break
        eigval = new_eigval
    return eigval, v, history


def power_iteration_deflated(matvec, dim, k, num_iters_per=300, seed=0):
    """Power iteration with rank-1 deflation to recover top-k eigenpairs."""
    g = torch.Generator().manual_seed(seed)
    found_vals, found_vecs = [], []

    def deflated_matvec(v):
        out = matvec(v)
        for lam, u in zip(found_vals, found_vecs):
            out = out - lam * (u @ v) * u
        return out

    for _ in range(k):
        v0 = torch.randn(dim, generator=g)
        v0 = v0 / v0.norm()
        v = v0
        eigval = 0.0
        for _ in range(num_iters_per):
            Av = deflated_matvec(v)
            new_eigval = (v @ Av).item()
            v = Av / (Av.norm() + 1e-30)
            eigval = new_eigval
        found_vals.append(eigval)
        found_vecs.append(v)
    return found_vals, found_vecs
