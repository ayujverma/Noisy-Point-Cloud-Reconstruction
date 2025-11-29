import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

# Cosine noise scheduler
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DiffusionScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        
        self.betas = cosine_beta_schedule(timesteps)
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        """
        Extract coefficients at specified timesteps t.
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# 1-D UNet Model for Point Clouds
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, n_groups=8):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        
        time_emb = self.mlp(time_emb)
        h = h + time_emb.unsqueeze(-1)
        
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, n_heads=4, n_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(n_groups, channels)
        self.attention = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        
    def forward(self, x):
        # x: (B, C, N) -> (B, N, C) for attention
        b, c, n = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 1)
        
        attn_out, _ = self.attention(h, h, h)
        
        # (B, N, C) -> (B, C, N)
        attn_out = attn_out.permute(0, 2, 1)
        return x + attn_out

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, channel_mults=(1, 2, 4, 8), time_emb_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Down path
        self.downs = nn.ModuleList()
        curr_channels = base_channels
        in_out = []
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            block = nn.ModuleList([
                ResidualBlock(curr_channels, out_channels, time_emb_dim),
                ResidualBlock(out_channels, out_channels, time_emb_dim),
                AttentionBlock(out_channels),
                Downsample(out_channels) if i != len(channel_mults) - 1 else nn.Identity()
            ])
            self.downs.append(block)
            in_out.append((curr_channels, out_channels))
            curr_channels = out_channels

        # Middle
        self.mid_block1 = ResidualBlock(curr_channels, curr_channels, time_emb_dim)
        self.mid_block2 = ResidualBlock(curr_channels, curr_channels, time_emb_dim)
        self.mid_attn = AttentionBlock(curr_channels)
        self.mid_block3 = ResidualBlock(curr_channels, curr_channels, time_emb_dim)
        self.mid_block4 = ResidualBlock(curr_channels, curr_channels, time_emb_dim)

        # Up path
        self.ups = nn.ModuleList()
        reversed_mults = list(reversed(channel_mults))
        skip_channels_stack = [x[1] for x in in_out]

        for i, mult in enumerate(reversed_mults):
            if i < len(reversed_mults) - 1:
                out_channels = base_channels * reversed_mults[i+1]
            else:
                out_channels = base_channels
            
            skip_ch = skip_channels_stack.pop()
            
            block = nn.ModuleList([
                Upsample(curr_channels) if i != 0 else nn.Identity(),
                ResidualBlock(curr_channels + skip_ch, out_channels, time_emb_dim),
                ResidualBlock(out_channels, out_channels, time_emb_dim),
                AttentionBlock(out_channels)
            ])
            self.ups.append(block)
            curr_channels = out_channels

        self.final_norm = nn.GroupNorm(8, curr_channels)
        self.final_conv = nn.Conv1d(curr_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        # x: (B, 3, N), time: (B,)
        t_emb = self.time_mlp(time)
        x = self.init_conv(x)
        skips = []

        # Down path
        for block in self.downs:
            # Apply all layers except the last one (Downsample/Identity)
            for layer in block[:-1]:
                if isinstance(layer, ResidualBlock):
                    x = layer(x, t_emb)
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
            
            # Save skip connection BEFORE downsampling
            skips.append(x)
            
            # Apply Downsample/Identity
            x = block[-1](x)

        # Middle
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block3(x, t_emb)
        x = self.mid_block4(x, t_emb)

        # Up path
        for block in self.ups:
            for layer in block:
                if isinstance(layer, Upsample):
                    x = layer(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    x = layer(x, t_emb)
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x

# Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        
    def forward(self, x_0):
        """
        Training step:
        1. Sample t
        2. Add noise
        3. Predict noise
        4. Return loss
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        t = torch.randint(0, self.scheduler.timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_0)
        
        x_t = self.scheduler.q_sample(x_0, t, noise)
        
        predicted_noise = self.model(x_t, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, batch_size, num_points, device):
        """
        Sampling loop (Reverse process)
        """
        # Start from pure noise
        img = torch.randn((batch_size, 3, num_points), device=device)
        
        for i in reversed(range(0, self.scheduler.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(img, t)
            
            alpha = self.scheduler.extract(self.scheduler.alphas, t, img.shape)
            alpha_cumprod = self.scheduler.extract(self.scheduler.alphas_cumprod, t, img.shape)
            beta = self.scheduler.extract(self.scheduler.betas, t, img.shape)
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            # DDPM
            inv_sqrt_alpha = 1.0 / torch.sqrt(alpha)
            noise_coeff = (1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod)
            
            mean = inv_sqrt_alpha * (img - noise_coeff * predicted_noise)
            
            # Variance
            posterior_variance = self.scheduler.extract(self.scheduler.posterior_variance, t, img.shape)
            sigma = torch.sqrt(posterior_variance)
            
            img = mean + sigma * noise
            
        return img

# Training Loop
def train(model, dataloader, optimizer, epochs=5, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            # batch shape: (B, N, 3) -> need (B, 3, N) for Conv1d
            batch = batch.permute(0, 2, 1)
            
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")