from diffusers import AutoencoderDC
import torch
import torch.nn.functional as F
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, IterableDataset
from PIL import Image
import random
import tqdm
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType
import os
import torchvision.utils as vutils

MODELS_DIR_BASE = "../../models"
AE_HF_NAME = "dc-ae-f32c32-sana-1.1-diffusers"

if __name__ == "__main__":
    os.makedirs("logs/recon", exist_ok=True)
    os.makedirs("logs/downsized_recon", exist_ok=True)
    os.makedirs("logs/latent_channels", exist_ok=True)

    # Load the autoencoder model
    ae = AutoencoderDC.from_pretrained(
        f"mit-han-lab/{AE_HF_NAME}",
        cache_dir=f"{MODELS_DIR_BASE}/dc_ae",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Freeze all but the last 2 encoder blocks and conv_out
    for i, block in enumerate(ae.encoder.down_blocks):
        if i < len(ae.encoder.down_blocks) - 1:  # Freeze all but the last 2 blocks
            for param in block.parameters():
                param.requires_grad = False

    for param in ae.encoder.conv_in.parameters():  # Freeze conv_in
        param.requires_grad = False

    # Keep conv_out trainable
    for param in ae.encoder.conv_out.parameters():
        param.requires_grad = True

    # Freeze all but the first decoder block and conv_in
    for i, block in enumerate(reversed(ae.decoder.up_blocks)):
        if i > 0:  # Freeze all but the first decoder block
            for param in block.parameters():
                param.requires_grad = False

    for param in ae.decoder.conv_out.parameters():  # Freeze conv_out
        param.requires_grad = False

    # Keep conv_in trainable
    for param in ae.decoder.conv_in.parameters():
        param.requires_grad = True

    # Count trainable and untrainable parameters
    total_params = sum(p.numel() for p in ae.parameters())
    trainable_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    untrainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Untrainable parameters: {untrainable_params:,}")

    def filter_small_images(row):
        return row["image"].height >= 512 or row["image"].width >= 512

    # ds = load_dataset("opendiffusionai/pexels-photos-janpf", split="train", streaming=True)
    ds = load_dataset("nelorth/oxford-flowers", split="train", cache_dir="../../datasets/oxford-flowers")
    ds = ds.filter(filter_small_images)

    transform = transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop((1024, 1024)),
        transforms.ToTensor(),
    ])

    class ImageDataset(IterableDataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __iter__(self):
            for row in iter(self.dataset):
                image = row["image"].convert("RGB")
                if self.transform:
                    image = self.transform(image)
                yield image  # You can return a tuple (image, label) if needed

    # Create dataset
    image_dataset = ImageDataset(ds, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(image_dataset, batch_size=1)

    convnext_loss = ConvNextPerceptualLoss(
        model_type=ConvNextType.TINY,
        feature_layers=[2,6,10,14],
        use_gram=False,
        input_range=(-1,1),
        device="cuda"
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ae.parameters()), lr=1e-5)

    for epoch in range(10):
        progress_bar = tqdm.tqdm(dataloader)
        for step, batch in enumerate(progress_bar):
            batch = batch.to("cuda").to(torch.bfloat16)
            # s = random.choice([160, 192, 224])
            s = random.choice([512, 768])
            downsized = F.interpolate(batch, (s, s))

            z = ae.encode(batch).latent
            s = s // 32
            downsized_z = F.interpolate(z, (s,s))

            recon = ae.decode(z).sample
            downsized_recon = ae.decode(downsized_z).sample

            l1_loss = F.l1_loss(recon, batch) + 0.25 * F.l1_loss(downsized_recon, downsized)
            perceptual_loss = convnext_loss(recon, batch) + 0.25 * convnext_loss(downsized_recon, downsized)

            loss = l1_loss + perceptual_loss * 0.3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

            if step % 500 == 0:
                with torch.no_grad():
                    def normalize_image(img):
                        img = img.float()  # Convert from bfloat16
                        img_min = img.amin(dim=(1,2,3), keepdim=True)
                        img_max = img.amax(dim=(1,2,3), keepdim=True)
                        img = (img - img_min) / (img_max - img_min + 1e-5)  # Normalize per image
                        return img

                    batch_vis = normalize_image(batch)
                    recon_vis = normalize_image(recon)
                    downsized_vis = normalize_image(downsized)
                    downsized_recon_vis = normalize_image(downsized_recon)

                    # Save image vs. recon
                    vutils.save_image(
                        torch.cat([batch_vis, recon_vis], dim=0),
                        f"logs/recon/step_{step}.png",
                        nrow=batch.shape[0]
                    )

                    # Save downsized vs. downsized_recon
                    vutils.save_image(
                        torch.cat([downsized_vis, downsized_recon_vis], dim=0),
                        f"logs/downsized_recon/step_{step}.png",
                        nrow=batch.shape[0]
                    )

                    # Extract channels 12, 13, 14 and normalize for RGB visualization
                    latent_rgb = z[:, 12:15, :, :]  # Select channels 12, 13, 14

                    # Normalize each channel independently to [0,1]
                    latent_rgb_min = latent_rgb.amin(dim=(2, 3), keepdim=True)
                    latent_rgb_max = latent_rgb.amax(dim=(2, 3), keepdim=True)
                    latent_rgb = (latent_rgb - latent_rgb_min) / (latent_rgb_max - latent_rgb_min + 1e-5)  # Avoid div by zero

                    # Save latents as an RGB image
                    vutils.save_image(
                        latent_rgb.float(),
                        f"logs/latent_channels/step_{step}.png",
                        nrow=batch.shape[0]
                    )