import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


class PosEnc(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        self.freq_bands = 2.0 ** torch.arange(L) * math.pi
    
    def forward(self, x):
        freq_bands = self.freq_bands.to(x.device)
        angles = x[..., None] * freq_bands
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        pe = pe.flatten(-2)
        return torch.cat([x, pe], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, width=256, pe_L=10):
        super().__init__()
        self.pe = PosEnc(L=pe_L)
        pe_dim = 2 + (2 * 2 * pe_L)
        self.layers = nn.Sequential(
            nn.Linear(pe_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(self.pe(x))


class ImagePixelDataLoader:
    def __init__(self, image_path, batch_size, device="cuda"):
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=np.uint8)
        
        self.H, self.W = img_array.shape[:2]
        
        self.rgb_normalized = torch.from_numpy(img_array).float() / 255.0
        self.rgb_normalized = self.rgb_normalized.to(device)
        
        self.batch_size = batch_size
        self.device = device
    
    def sample_batch(self):
        ys = torch.randint(0, self.H, (self.batch_size,), device=self.device)
        xs = torch.randint(0, self.W, (self.batch_size,), device=self.device)
        
        x_coords = xs.float() / float(self.W)
        y_coords = ys.float() / float(self.H)
        coords = torch.stack([x_coords, y_coords], dim=-1)
        
        colors = self.rgb_normalized[ys, xs, :]
        
        return coords, colors
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample_batch()


def psnr_from_mse(mse):
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def train(model, dataloader, num_iterations=2000, lr=1e-2, device="cuda", log_every=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    model.train()
    for iteration in range(1, num_iterations + 1):
        coords, target_colors = dataloader.sample_batch()
        
        pred_colors = model(coords)
        loss = loss_fn(pred_colors, target_colors)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % log_every == 0 or iteration == 1:
            with torch.no_grad():
                psnr = psnr_from_mse(loss)
            print(f"[{iteration:5d}/{num_iterations}] Loss={loss.item():.6f}  PSNR={psnr.item():.2f} dB")
            
            
            # if not hasattr(train, "psnr_hist"):
            #     train.psnr_hist = []
            #     train.iter_hist = []

            # train.psnr_hist.append(psnr.item())
            # train.iter_hist.append(iteration)

            # if iteration == num_iterations:
            #     import matplotlib.pyplot as plt
            #     plt.figure()
            #     plt.plot(train.iter_hist, train.psnr_hist, marker='o')
            #     plt.xlabel('Iteration')
            #     plt.ylabel('PSNR (dB)')
            #     plt.title('PSNR Curve During Training')
            #     plt.grid(True)
            #     plt.savefig("psnr_curve.png")
            #     plt.close()
            # if iteration in [10, 50, 100, 200, 500, 1000, 1500, 2000]:
            #     with torch.no_grad():
            #         model.eval()
            #         rgb_normalized = render_image(model, dataloader, device)
            #         rgb_normalized = rgb_normalized.cpu().numpy()
            #         rgb_normalized = (rgb_normalized * 255.0).astype(np.uint8)
            #         rgb_normalized = Image.fromarray(rgb_normalized)
            #         rgb_normalized.save(f"kite_reconstruction_{iteration:05d}.png")
    return model

def render_image(model, dataloader, device, tile_size=262144):
    H, W = dataloader.H, dataloader.W
    model.eval()
    
    with torch.no_grad():
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        x_coords = xx.float() / float(W)
        y_coords = yy.float() / float(H)
        coords = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
        
        output = []
        for i in range(0, coords.shape[0], tile_size):
            chunk = coords[i:i+tile_size]
            pred_colors = model(chunk)
            output.append(pred_colors)
        
        rgb_normalized = torch.cat(output, dim=0).reshape(H, W, 3).clamp(0, 1)
        return rgb_normalized


if __name__ == "__main__":
    image_path = "media/wolf/wolf.jpg"
    batch_size = 10000
    device = "cpu"
    num_iterations = 2000
    lr = 1e-2
    log_every = 100
    pe_L = 2
    width = 64
    
    model = MLP(in_dim=2, out_dim=3, width=width, pe_L=pe_L)
    dataloader = ImagePixelDataLoader(image_path, batch_size, device)
    model = train(model, dataloader, num_iterations, lr, device, log_every)

    with torch.no_grad():
        image = render_image(model, dataloader, device)
        image = image.cpu().numpy()
        image = (image * 255.0).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(f"wolf_reconstruction_width_{width}_pe_{pe_L}.png")