import torch
from transformer.meikai import create_models, train_hvit_vae
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from torchvision import transforms

if __name__ == "__main__":
    # Parameters
    image_size = 256  # The CelebA-HQ dataset already has 256x256 images
    batch_size = 8
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    vae, discriminator = create_models(image_size=image_size, latent_dim=32, dim_mults=(4, 4, 8, 8))

    # Print model parameter counts
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())}")
    
    vae = vae.to(device)
    discriminator = discriminator.to(device)
    
    # Optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-4, betas=(0.5, 0.9))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
    
    # Load CelebA-HQ dataset from Hugging Face
    print("Loading CelebA-HQ dataset...")
    dataset = load_dataset("korexyz/celeba-hq-256x256", cache_dir="data", split="train")
    
    # Define transformation to convert HF dataset images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL images to tensors (0-1 range)
    ])
    
    # Create a custom dataset class to apply transforms
    class CelebaHQDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            # Get image from the dataset
            img = self.dataset[idx]['image']
            
            # Apply transform if specified
            if self.transform:
                img = self.transform(img)
                
            # Return image and a dummy label (not used in training)
            return img, 0
    
    # Create custom dataset with transforms
    custom_dataset = CelebaHQDataset(dataset, transform=transform)
    
    # Create dataloader
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset loaded with {len(custom_dataset)} images")
    
    # Training
    vae, discriminator = train_hvit_vae(
        vae, 
        discriminator, 
        vae_optimizer, 
        disc_optimizer, 
        dataloader,
        num_epochs,
        device,
        kld_weight=0.01,
        adv_weight=0.1,
        perceptual_weight=1.0,
        gp_weight=100.0,
        image_log_interval=100,
        log_dir='logs/celeba_hq'
    )
    
    # Save models
    torch.save(vae.state_dict(), 'hvit_vae_celeba_hq.pth')
    torch.save(discriminator.state_dict(), 'hvit_discriminator_celeba_hq.pth')