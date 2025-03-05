from diffusers import AutoencoderDC
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    # Load the autoencoder model
    ae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        cache_dir="../models",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Freeze all but the last 2 encoder blocks and conv_out
    for i, block in enumerate(ae.encoder.down_blocks):
        if i < len(ae.encoder.down_blocks) - 2:  # Freeze all but the last 2 blocks
            for param in block.parameters():
                param.requires_grad = False

    for param in ae.encoder.conv_in.parameters():  # Freeze conv_in
        param.requires_grad = False

    # Keep conv_out trainable
    for param in ae.encoder.conv_out.parameters():
        param.requires_grad = True

    # Freeze all but the first decoder block and conv_in
    for i, block in enumerate(ae.decoder.up_blocks):
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

    