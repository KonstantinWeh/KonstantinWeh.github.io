import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from volume_rendering import volrend
from nerf_multi_view import (
    RaysData,
    pixel_to_ray,
    sample_points_along_rays,
    images_train,
    images_val,
    c2ws_train,
    c2ws_test,
    c2ws_val,
    focal,
)

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional for rendering
    imageio = None


@dataclass
class TrainingConfig:
    num_iterations: int = 10
    batch_size: int = 1_024
    n_samples: int = 64
    near: float = 2.0
    far: float = 6.0
    lr: float = 5e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 2
    chunk_size: int = 2_048
    save_path: Optional[str] = None
    load_path: Optional[str] = None


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_freqs, include_input=True):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.include_input = include_input

        freq_bands = 2.0 ** torch.arange(num_freqs) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_dim(self):
        base = 2 * self.in_dim * self.num_freqs
        return base + (self.in_dim if self.include_input else 0)

    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {x.shape[-1]}")

        raw = x
        angles = raw.unsqueeze(-1) * self.freq_bands
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-2)
        enc = enc.flatten(-2)

        if self.include_input:
            enc = torch.cat([raw, enc], dim=-1)

        return enc


class NeRFMLP(nn.Module):
    def __init__(self, width=256, num_layers=8, skips=(4,), pos_freqs=10, dir_freqs=4):
        super().__init__()

        self.pos_enc = PositionalEncoding(in_dim=3, num_freqs=pos_freqs, include_input=True)
        self.dir_enc = PositionalEncoding(in_dim=3, num_freqs=dir_freqs, include_input=True)

        self.width = width
        self.num_layers = num_layers
        self.skips = set(skips)

        pos_dim = self.pos_enc.out_dim
        self.pts_linears = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = pos_dim
            elif i in self.skips:
                in_dim = width + pos_dim
            else:
                in_dim = width
            self.pts_linears.append(nn.Linear(in_dim, width))

        self.sigma_linear = nn.Linear(width, 1)
        self.feature_linear = nn.Linear(width, width)

        dir_dim = self.dir_enc.out_dim
        self.rgb_layers = nn.Sequential(
            nn.Linear(width + dir_dim, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = self.pos_enc(x)
        h = x_enc
        for i, layer in enumerate(self.pts_linears):
            if i in self.skips and i != 0:
                h = torch.cat([h, x_enc], dim=-1)
            h = layer(h)
            h = F.relu(h, inplace=True)

        sigma = F.softplus(self.sigma_linear(h))

        features = self.feature_linear(h)

        d_norm = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-8)
        d_enc = self.dir_enc(d_norm)
        color_input = torch.cat([features, d_enc], dim=-1)
        rgb = self.rgb_layers(color_input)

        return rgb, sigma

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    losses: Sequence[float],
    step_size: float,
    iteration: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "losses": list(losses),
        "step_size": float(step_size),
        "iteration": int(iteration),
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    checkpoint = torch.load(path, map_location=device if device is not None else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_nerf(config: TrainingConfig) -> dict:
    device = torch.device(config.device)

    model = NeRFMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    # camera intrinsics for dataset
    h, w = images_train.shape[1:3]
    focal_val = float(focal)
    k = torch.tensor(
        [
            [focal_val, 0.0, w / 2.0],
            [0.0, focal_val, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    dataset = RaysData(images_train, k.cpu().numpy(), c2ws_train)

    step_size = (config.far - config.near) / config.n_samples

    losses = []
    start_iter = 0
    if config.load_path:
        checkpoint = load_checkpoint(config.load_path, model, optimizer, device)
        losses = list(checkpoint.get("losses", []))
        start_iter = int(checkpoint.get("iteration", 0))
        print(f"Loaded checkpoint from {config.load_path} (iteration {start_iter})")

    for it in range(start_iter + 1, start_iter + config.num_iterations + 1):
        rays_o, rays_d, pixels = dataset.sample_rays(config.batch_size)

        points, _ = sample_points_along_rays(
            rays_o,
            rays_d,
            near=config.near,
            far=config.far,
            n_samples=config.n_samples,
            perturb=True,
        )

        rays_o_t = torch.from_numpy(rays_o).to(device=device, dtype=torch.float32)
        rays_d_t = torch.from_numpy(rays_d).to(device=device, dtype=torch.float32)
        target_pixels = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
        points_t = torch.from_numpy(points).to(device=device, dtype=torch.float32)

        bsz, n_samples, _ = points_t.shape
        directions = rays_d_t[:, None, :].expand(-1, n_samples, -1)

        points_flat = points_t.reshape(-1, 3)
        dirs_flat = directions.reshape(-1, 3)

        rendered_rgb, rendered_sigma = model(points_flat, dirs_flat)

        rgbs = rendered_rgb.view(bsz, n_samples, 3)
        sigmas = rendered_sigma.view(bsz, n_samples, 1)

        rendered = volrend(sigmas, rgbs, step_size)

        loss = criterion(rendered, target_pixels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if config.log_every and (it % config.log_every == 0):
            span = losses[-config.log_every:]
            avg_loss = sum(span) / len(span)
            print(f"Iter {it:05d}: loss={avg_loss:.6f}")
        
        val_psnrs = []
        
        # Try to infer validation set based on c2ws_test
        with torch.no_grad():
            model.eval()
            # Use the training intrinsics and images' shape
            h_val, w_val = images_train.shape[1:3]
            for idx, c2w_val in enumerate(c2ws_val):
                rays_o_val, rays_d_val = _generate_camera_rays(k.cpu().numpy(), c2w_val, h_val, w_val)
                rendered_val = render_rays(model, rays_o_val, rays_d_val, config, step_size)
                rendered_val = rendered_val.reshape(h_val, w_val, 3)

                if idx == 0 and imageio is not None:
                    img_save = (np.clip(rendered_val, 0, 1) * 255).astype(np.uint8)
                    imageio.imwrite(f"val_iter_{it}.png", img_save)

                # Get GT image for this validation camera index if available, else skip
                if idx < images_val.shape[0]:
                    gt = images_val[idx]
                    # Resize or crop if needed
                    if gt.shape != rendered_val.shape:
                        min_h = min(gt.shape[0], rendered_val.shape[0])
                        min_w = min(gt.shape[1], rendered_val.shape[1])
                        gt = gt[:min_h, :min_w]
                        rendered_val = rendered_val[:min_h, :min_w]
                    mse = np.mean((rendered_val - gt) ** 2)
                    psnr = -10 * np.log10(mse + 1e-8)
                    val_psnrs.append(psnr)
    # Plot after training
    if val_psnrs:
        with open("validation_psnr_raw.txt", "w") as f:
            for psnr in val_psnrs:
                f.write(f"{psnr}\n")

    final_iter = start_iter + config.num_iterations

    if config.save_path:
        save_checkpoint(
            config.save_path,
            model,
            optimizer,
            config,
            losses,
            step_size,
            final_iter,
        )
        print(f"Saved checkpoint to {config.save_path}")

    return {
        "model": model,
        "losses": losses,
        "step_size": step_size,
        "config": config,
        "intrinsics": k,
    }


def _generate_camera_rays(k: np.ndarray, c2w: np.ndarray, height: int, width: int):
    uu, vv = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    uv = np.stack([uu + 0.5, vv + 0.5], axis=-1).reshape(-1, 2)

    c2w_rep = np.repeat(c2w[None, ...], uv.shape[0], axis=0)

    ray_o, ray_d = pixel_to_ray(k, c2w_rep, uv)

    return ray_o.astype(np.float32), ray_d.astype(np.float32)


def render_rays(
    model: nn.Module,
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    config: TrainingConfig,
    step_size: float,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device

    rendered_chunks = []
    n_rays = rays_o.shape[0]

    with torch.no_grad():
        for start in range(0, n_rays, config.chunk_size):
            end = min(start + config.chunk_size, n_rays)
            rays_o_batch = rays_o[start:end]
            rays_d_batch = rays_d[start:end]

            points_np, _ = sample_points_along_rays(
                rays_o_batch,
                rays_d_batch,
                near=config.near,
                far=config.far,
                n_samples=config.n_samples,
                perturb=False,
            )

            points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
            dirs = torch.from_numpy(rays_d_batch).to(device=device, dtype=torch.float32)
            dirs = dirs[:, None, :].expand(-1, config.n_samples, -1)

            points_flat = points.reshape(-1, 3)
            dirs_flat = dirs.reshape(-1, 3)

            rgbs, sigmas = model(points_flat, dirs_flat)
            rgbs = rgbs.view(-1, config.n_samples, 3)
            sigmas = sigmas.view(-1, config.n_samples, 1)

            colors = volrend(sigmas, rgbs, step_size)
            rendered_chunks.append(colors.cpu().numpy())

    return np.concatenate(rendered_chunks, axis=0)


def render_image(
    model: nn.Module,
    c2w: np.ndarray,
    k: np.ndarray,
    config: TrainingConfig,
    step_size: float,
) -> np.ndarray:
    height, width = images_train.shape[1:3]
    rays_o, rays_d = _generate_camera_rays(k, c2w, height, width)

    colors = render_rays(model, rays_o, rays_d, config, step_size)
    image = colors.reshape(height, width, 3)

    return np.clip(image, 0.0, 1.0)


def render_spherical_video(
    model: nn.Module,
    config: TrainingConfig,
    step_size: float,
    output_path: str,
    fps: int = 15,
):
    if imageio is None:
        raise ImportError("imageio is required to export the rendering video")

    height, width = images_train.shape[1:3]
    focal_val = float(focal)
    k = np.array(
        [
            [focal_val, 0.0, width / 2.0],
            [0.0, focal_val, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    frames = []
    for idx, c2w in enumerate(c2ws_test[:1]):
        print(f"Rendering test camera {idx + 1}/{len(c2ws_test[:1])}", end="\r")
        frame = render_image(model, c2w, k, config, step_size)
        plt.imshow(frame)
        plt.show()
        frames.append((frame * 255.0).astype(np.uint8))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(output_path, frames)
    print(f"\nSaved spherical video to {output_path}")


if __name__ == "__main__":
    cfg = TrainingConfig(load_path="checkpoints/lego_500.pth")
    result = train_nerf(cfg)
    print("Training complete. Final loss:", result["losses"][-1])
    if imageio is None:
        print("Install imageio to export the spherical rendering video.")
    else:
        output_file = Path("renders/lego_spherical.mp4")
        render_spherical_video(
            result["model"],
            cfg,
            result["step_size"],
            output_file,
        )

    