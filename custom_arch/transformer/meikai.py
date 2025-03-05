import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from relativistic_loss.loss import gan_loss, approximate_r1_loss, approximate_r2_loss
from torchvision.models import vgg16
import lpips
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from torchvision import transforms
from .layers import GlobalAttentionBlock, NeighborhoodAttentionBlock
from .utils import DownsampleBlock, UpsampleBlock
from .embed import RoPE2D

class HourglassEncoderLevel(nn.Module):
    def __init__(self, dim, head_dim, kernel_size, depth, is_global=False, dropout=0.1):
        super().__init__()
        self.is_global = is_global
        
        if is_global:
            self.blocks = nn.ModuleList([
                GlobalAttentionBlock(dim, head_dim, dropout=dropout)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                NeighborhoodAttentionBlock(dim, head_dim, kernel_size, dropout=dropout)
                for _ in range(depth)
            ])
                        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        return x

class HourglassDecoderLevel(nn.Module):
    def __init__(self, dim, head_dim, kernel_size, depth, is_global=False, dropout=0.1):
        super().__init__()
        self.is_global = is_global

        if is_global:
            self.blocks = nn.ModuleList([
                GlobalAttentionBlock(dim, head_dim, dropout=dropout)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                NeighborhoodAttentionBlock(dim, head_dim, kernel_size, dropout=dropout)
                for _ in range(depth)
            ])
            
    def forward(self, x):
        # x shape: [batch, height, width, channels]
        
        for block in self.blocks:
            x = block(x)
            
        return x

class HViTEncoder(nn.Module):
    def __init__(
        self,
        image_size=256,
        input_channels=3,
        patch_size=4,
        latent_dim=16,  # Dimension of latent space
        base_dim=128,
        dim_mults=(1, 2, 4, 8),
        depths=(2, 2, 2, 2),
        head_dim=64,
        kernel_size=7,
        dropout=0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.patched_size = image_size // patch_size
        self.latent_size = self.patched_size // (2 ** (len(dim_mults) - 1))
        self.latent_channels = latent_dim
        
        # Encoder levels
        enc_dims = [base_dim * m for m in dim_mults]

        # Initial patch embedding using PixelUnshuffle instead of Conv2D
        self.patch_embed = DownsampleBlock(input_channels, enc_dims[0], shortcut=False, factor=patch_size)
        self.norm = nn.LayerNorm([enc_dims[0]])
        
        self.encoder_levels = nn.ModuleList()
        
        # Build encoder downsampling path
        for i in range(0, len(dim_mults)):
            is_global = (i == len(dim_mults) - 1)  # Global attention at lowest level
            
            self.encoder_levels.append(
                HourglassEncoderLevel(
                    dim=enc_dims[i], head_dim=head_dim, kernel_size=kernel_size,
                    depth=depths[i], is_global=is_global, dropout=dropout
                )
            )

            if not is_global:
                self.encoder_levels.append(
                    DownsampleBlock(enc_dims[i], enc_dims[i+1], shortcut=False)
                )
        
        # Latent projections
        self.latent_mean = nn.Linear(enc_dims[-1], self.latent_channels)
        
    def forward(self, x):
        # x shape: (b, c, h, w)
        b, c, h, w = x.shape
        
        # (b, c, h, w) -> (b, h, w, c)
        x = x.permute(0, 2, 3, 1)  
        # (b, h, w, c) -> (b, h/p, w/p, base_dim)
        x = self.patch_embed(x)
        x = self.norm(x)             
                
        # Encoder path with downsampling
        for enc in self.encoder_levels:
            x = enc(x)
        
        # Project to latent space
        z = self.latent_mean(x)
        
        return z

class HViTDecoder(nn.Module):
    def __init__(
        self,
        image_size=256,
        output_channels=3,
        patch_size=4,
        latent_dim=16,  # Dimension of latent space
        base_dim=128,
        dim_mults=(8, 4, 2, 1),
        depths=(2, 2, 2, 2),
        head_dim=64,
        kernel_size=7,
        dropout=0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.patched_size = image_size // patch_size
        self.latent_size = self.patched_size // (2 ** (len(dim_mults) - 1))
        self.latent_channels = latent_dim
        
        # Decoder levels
        dec_dims = [base_dim * m for m in dim_mults]
        
        # Initial latent projection
        self.latent_proj = nn.Linear(self.latent_channels, dec_dims[0])
        
        # Decoder levels
        self.decoder_levels = nn.ModuleList()

        for i in range(0, len(dim_mults)):
            is_global = (i == 0)  # Global attention at lowest level

            if not is_global:
                self.decoder_levels.append(
                    UpsampleBlock(dec_dims[i-1], dec_dims[i], shortcut=False)
                )

            self.decoder_levels.append(
                HourglassDecoderLevel(dec_dims[i], head_dim, kernel_size, depths[i], is_global=is_global, dropout=dropout)
            )

        self.output_patch_embed = UpsampleBlock(dec_dims[-1], output_channels, shortcut=False, factor=patch_size)
        
    def forward(self, z):
        # z shape: [batch, latent_height, latent_width, latent_channels]
        
        # Initial projection from latent
        x = self.latent_proj(z)
                
        # Decoder path with upsampling
        for dec in self.decoder_levels:
            x = dec(x)
        
        # Final projection
        x = self.output_patch_embed(x)
        
        # Back to NCHW
        x = x.permute(0, 3, 1, 2)
        
        return torch.tanh(x)  # Scale output to [-1, 1]


class HViTVAE(nn.Module):
    def __init__(
        self,
        image_size=256,
        channels=3,
        patch_size=4,
        latent_dim=16,
        base_dim=128,
        dim_mults=(1, 2, 4, 8),
        depths=(2, 2, 2, 2),
        head_dim=64,
        kernel_size=7,
        dropout=0.1,
    ):
        super().__init__()
        
        self.encoder = HViTEncoder(
            image_size=image_size,
            input_channels=channels,
            patch_size=patch_size,
            latent_dim=latent_dim,
            base_dim=base_dim,
            dim_mults=dim_mults,
            depths=depths,
            head_dim=head_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        self.decoder = HViTDecoder(
            image_size=image_size,
            output_channels=channels,
            patch_size=patch_size,
            latent_dim=latent_dim,
            base_dim=base_dim,
            dim_mults=dim_mults[::-1],
            depths=depths[::-1],
            head_dim=head_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        # LPIPS loss for perceptual similarity
        self.perceptual_loss = lpips.LPIPS(net='vgg', spatial=False)
        
    def forward(self, x):
        z = self.encoder(x)
        # print(z.shape)
        recon = self.decoder(z)
        return recon
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        # For decoding from a latent without skip connections
        batch_size = z.shape[0]
        latent_size = self.encoder.latent_size
        
        # Reshape if needed
        if len(z.shape) == 2:
            z = z.reshape(batch_size, latent_size, latent_size, -1)
            
        return self.decoder(z)
    
class HViTDiscriminator(nn.Module):
    def __init__(
        self,
        image_size=256,
        input_channels=3,
        patch_size=4,
        base_dim=128,
        dim_mults=(1, 2, 4, 8),
        depths=(2, 2, 2, 2),
        head_dim=64,
        kernel_size=7,
        dropout=0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.patched_size = image_size // patch_size
        self.latent_size = self.patched_size // (2 ** (len(dim_mults) - 1))
        
        # Encoder levels
        enc_dims = [base_dim * m for m in dim_mults]

        # Initial patch embedding using PixelUnshuffle instead of Conv2D
        self.patch_embed = DownsampleBlock(input_channels, enc_dims[0], shortcut=False, factor=patch_size)
        self.norm = nn.LayerNorm([enc_dims[0]])

        self.cond_proj = nn.Linear(input_channels, enc_dims[0])
        
        self.encoder_levels = nn.ModuleList()
        
        # Build encoder downsampling path
        for i in range(0, len(dim_mults)):
            is_global = (i == len(dim_mults) - 1)  # Global attention at lowest level
            
            self.encoder_levels.append(
                HourglassEncoderLevel(
                    dim=enc_dims[i], head_dim=head_dim, kernel_size=kernel_size,
                    depth=depths[i], is_global=is_global, dropout=dropout
                )
            )

            if not is_global:
                self.encoder_levels.append(
                    DownsampleBlock(enc_dims[i], enc_dims[i+1], shortcut=False)
                )
        
        # Output projection
        self.logit_proj = nn.Linear(enc_dims[-1], 1)
        
    def forward(self, x, cond):
        # x shape: (b, c, h, w)
        # cond shape: (b, c, h // 4, w // 4) 
        # cond is 4x downsampled version of x
        b, c, h, w = x.shape
        
        # (b, c, h, w) -> (b, h, w, c)
        x = x.permute(0, 2, 3, 1)  
        # (b, h, w, c) -> (b, h/p, w/p, base_dim)
        x = self.patch_embed(x)
        x = self.norm(x)

        cond = cond.permute(0, 2, 3, 1)
        cond = self.cond_proj(cond)
        x = x + cond
                
        # Encoder path with downsampling
        for enc in self.encoder_levels:
            x = enc(x)
        
        # Project to latent space
        logits = self.logit_proj(x)
        
        return logits

def compute_vae_loss(
    recon, 
    x, 
    discriminator, 
    kld_weight=0.01, 
    adv_weight=0.5, 
    perceptual_weight=1.0,
    perceptual_loss_fn=None
):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x)
    recon_loss = recon_loss + F.l1_loss(recon, x)
    
    # KL divergence loss
    # kld_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    # cond is 4x downsampled version of x
    cond = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
    
    # Adversarial loss - use our generator to fool the discriminator
    # Instead of passing fake_data directly, we create a simple generator function
    def vae_as_generator(z):
        return recon
        
    adv_loss = gan_loss(
        discriminator, vae_as_generator,
        x, None,  # Real data and latent input (not used by our generator)
        discriminator_turn=False,
        disc_kwargs={'cond': cond}
    )
    
    # Perceptual loss using LPIPS
    if perceptual_loss_fn is not None:
        # Rescale for LPIPS which expects images in [-1, 1]
        x_scaled = (x * 2) - 1
        recon_scaled = (recon * 2) - 1
        perceptual_loss = perceptual_loss_fn(x_scaled, recon_scaled).mean()
    else:
        perceptual_loss = torch.tensor(0.0, device=x.device)
    
    # Total loss
    total_loss = recon_loss
    # total_loss +=  kld_weight * kld_loss 
    total_loss += adv_weight * adv_loss
    total_loss += perceptual_weight * perceptual_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        # 'kld': kld_loss.item(),
        'adv': adv_loss.item(),
        'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss
    }


def discriminator_loss(discriminator, vae, real_images, adv_weight=1.0, gp_weight=10.0):
    with torch.no_grad():
        # Generate fake images using the VAE
        fake_images = vae(real_images)
    
    # Relativistic GAN loss
    # Define a simple generator function that returns the pre-computed fake images
    def fake_generator(z):
        return fake_images
    
    # cond is 4x downsampled version of x
    cond = F.interpolate(real_images, scale_factor=0.25, mode='bilinear', align_corners=False)
        
    rel_loss = gan_loss(
        discriminator, fake_generator,
        real_images, None,  # Real data and latent input (not used by our generator)
        discriminator_turn=True,
        disc_kwargs={'cond': cond}
    )
    
    # Approximate R1 gradient penalty (on real data)
    r1_penalty = approximate_r1_loss(discriminator, real_images, sigma=0.01, Lambda=gp_weight, disc_kwargs={'cond': cond})
    
    # Approximate R2 gradient penalty (on fake data)
    r2_penalty = approximate_r2_loss(discriminator, fake_images, sigma=0.01, Lambda=gp_weight, disc_kwargs={'cond': cond})
    
    # Total discriminator loss
    total_loss = adv_weight * rel_loss + r1_penalty + r2_penalty
    
    return total_loss, {
        'total': total_loss.item(),
        'rel_loss': rel_loss.item(),
        'r1_penalty': r1_penalty.item(),
        'r2_penalty': r2_penalty.item()
    }


# Training function for the full HViT VAE model
def train_hvit_vae(
    vae, 
    discriminator, 
    vae_optimizer, 
    disc_optimizer, 
    dataloader, 
    num_epochs, 
    device,
    kld_weight=0.01,
    adv_weight=0.5,
    perceptual_weight=1.0,
    gp_weight=10.0,
    image_log_interval=200,
    log_dir='logs/images'
):
    # Initialize LPIPS loss
    perceptual_loss_fn = lpips.LPIPS(net='vgg', spatial=False).to(device)
    
    # Create directory for saving images
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        vae.train()
        discriminator.train()
        
        total_vae_loss = 0
        total_disc_loss = 0
        
        # Initialize tqdm progress bar
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                   desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (data, _) in pbar:
            data = data.to(device)
            
            # Step 1: Train Discriminator
            disc_optimizer.zero_grad()
            
            disc_loss, disc_loss_dict = discriminator_loss(
                discriminator, vae, data, 
                adv_weight=adv_weight, 
                gp_weight=gp_weight
            )
            
            disc_loss.backward()
            disc_optimizer.step()
            
            # Step 2: Train VAE
            vae_optimizer.zero_grad()
            
            recon = vae(data)
            
            vae_loss, vae_loss_dict = compute_vae_loss(
                recon, data, discriminator,
                kld_weight=kld_weight,
                adv_weight=adv_weight,
                perceptual_weight=perceptual_weight,
                perceptual_loss_fn=perceptual_loss_fn
            )
            
            vae_loss.backward()
            vae_optimizer.step()
            
            # Update total loss tracking
            total_vae_loss += vae_loss.item()
            total_disc_loss += disc_loss.item()
            
            # Log images and reconstructions every image_log_interval batches
            if batch_idx % image_log_interval == 0:
                # Save max 2 images from the batch
                num_images = min(2, data.size(0))
                comparison = torch.cat([
                    data[:num_images],  # Original images
                    recon[:num_images]   # Reconstructed images
                ])
                
                # Save images
                save_image(
                    comparison,
                    log_path / f'reconstruction_epoch_{epoch+1}_batch_{batch_idx}.png',
                    nrow=num_images
                )
                print(f"Saved image reconstructions at epoch {epoch+1}, batch {batch_idx}")
            
            # Update tqdm progress bar with current losses
            pbar.set_postfix({
                'VAE Loss': f"{vae_loss.item():.4f}",
                'Disc Loss': f"{disc_loss.item():.4f}",
                'Recon': f"{vae_loss_dict['recon']:.4f}",
                # 'KLD': f"{vae_loss_dict['kld']:.4f}",
                'Adv': f"{vae_loss_dict['adv']:.4f}",
                'Perceptual': f"{vae_loss_dict['perceptual']:.4f}"
            })
                
        # End of epoch summary
        avg_vae_loss = total_vae_loss / len(dataloader)
        avg_disc_loss = total_disc_loss / len(dataloader)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Avg VAE Loss: {avg_vae_loss:.4f}, Avg Disc Loss: {avg_disc_loss:.4f}")
    
    return vae, discriminator

def create_models(
    image_size=256,
    channels=3,
    patch_size=4,
    latent_dim=16,
    base_dim=128,
    dim_mults=(1, 2, 4, 8),
    depths=(2, 2, 2, 2),
    head_dim=64,
    kernel_size=7,
    dropout=0.1,
):
    """
    Create the HViT VAE and Discriminator models
    
    Args:
        image_size: Input image size (assumed square)
        channels: Number of image channels
        patch_size: Size of patches for transformer
        latent_dim: Dimension for latent space
        base_dim: Base dimension for model width
        dim_mults: Dimension multipliers for each level
        depths: Number of transformer blocks at each level
        heads: Number of attention heads at each level
        head_dim: Dimension of each attention head
        kernel_size: Kernel size for neighborhood attention
        dropout: Dropout rate
        
    Returns:
        vae: The HViT VAE model
        discriminator: The HViT Discriminator model
    """
    # Create VAE model
    vae = HViTVAE(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        latent_dim=latent_dim,
        base_dim=base_dim,
        dim_mults=dim_mults,
        depths=depths,
        head_dim=head_dim,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    
    # Create Discriminator model
    discriminator = HViTDiscriminator(
        image_size=image_size,
        input_channels=channels,
        patch_size=patch_size,
        base_dim=base_dim,
        dim_mults=dim_mults,
        depths=depths,
        head_dim=head_dim,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    
    return vae, discriminator

# Example usage
