"""Volume rendering utilities for NeRF."""

import torch


def volrend(sigmas: torch.Tensor, rgbs: torch.Tensor, step_size: float) -> torch.Tensor:
    """Render colors along rays via discrete volume rendering.

    Args:
        sigmas: Density values per sample with shape ``(B, N, 1)``.
        rgbs: Color values per sample with shape ``(B, N, 3)``.
        step_size: Distance between consecutive samples along each ray.

    Returns:
        Tensor of shape ``(B, 3)`` containing the rendered color for each ray.
    """

    if sigmas.dim() != 3 or sigmas.size(-1) != 1:
        raise ValueError("sigmas must have shape (B, N, 1)")
    if rgbs.dim() != 3 or rgbs.size(-1) != 3:
        raise ValueError("rgbs must have shape (B, N, 3)")
    if sigmas.shape[:2] != rgbs.shape[:2]:
        raise ValueError("sigmas and rgbs must agree on the first two dimensions")

    sigma_delta = sigmas * step_size
    alphas = 1.0 - torch.exp(-sigma_delta)

    cumulative_sigma = torch.cumsum(sigma_delta, dim=1)
    transmittance = torch.exp(-(cumulative_sigma - sigma_delta))

    weights = transmittance * alphas

    rendered = torch.sum(weights * rgbs, dim=1)

    return rendered

if __name__ == "__main__":
    torch.manual_seed(42)
    sigmas = torch.rand((10, 64, 1))
    rgbs = torch.rand((10, 64, 3))
    step_size = (6.0 - 2.0) / 64
    rendered_colors = volrend(sigmas, rgbs, step_size)

    correct = torch.tensor([
        [0.5006, 0.3728, 0.4728],
        [0.4322, 0.3559, 0.4134],
        [0.4027, 0.4394, 0.4610],
        [0.4514, 0.3829, 0.4196],
        [0.4002, 0.4599, 0.4103],
        [0.4471, 0.4044, 0.4069],
        [0.4285, 0.4072, 0.3777],
        [0.4152, 0.4190, 0.4361],
        [0.4051, 0.3651, 0.3969],
        [0.3253, 0.3587, 0.4215]
    ])
    assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)
    print("Test passed!")