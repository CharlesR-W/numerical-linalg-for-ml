import torch

from solutions._01_krylov import power_iteration, power_iteration_deflated


def test_power_iteration_top_eigval():
    torch.manual_seed(0)
    A = torch.diag(torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5]))
    Q = torch.linalg.qr(torch.randn(5, 5))[0]
    A = Q @ A @ Q.T  # spectrum {5,3,2,1,0.5} in a non-canonical basis

    def matvec(v):
        return A @ v

    eigval, vec, history = power_iteration(matvec, dim=5, num_iters=200, seed=0)
    assert abs(eigval - 5.0) < 1e-4
    # convergence history should be monotone non-increasing (mostly)
    assert history[-1] < history[10]


def test_deflation_top3():
    torch.manual_seed(0)
    A = torch.diag(torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5]))
    Q = torch.linalg.qr(torch.randn(5, 5))[0]
    A = Q @ A @ Q.T

    def matvec(v):
        return A @ v

    eigvals, _ = power_iteration_deflated(
        matvec, dim=5, k=3, num_iters_per=400, seed=0
    )
    expected = torch.tensor([5.0, 3.0, 2.0])
    assert torch.allclose(torch.tensor(eigvals), expected, atol=1e-3)
