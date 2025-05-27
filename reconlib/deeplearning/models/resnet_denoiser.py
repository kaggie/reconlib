import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    """
    A simple ResNet block with two convolutional layers.
    Input and output are expected to have the same number of channels.
    """
    def __init__(self, num_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual # Add skip connection
        out = self.relu(out)
        return out

class SimpleResNetDenoiser(nn.Module):
    """
    A simple ResNet-based denoiser for MRI reconstruction.
    Assumes input is a 2D image (e.g., single slice or processed multi-channel slice).
    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, out_channels, height, width)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_internal_channels: int = 64, num_blocks: int = 5, kernel_size: int = 3):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(in_channels, num_internal_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_internal_channels)
        self.relu_initial = nn.ReLU(inplace=True)
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResNetBlock(num_internal_channels, kernel_size))
        self.resnet_blocks = nn.Sequential(*blocks)
        
        self.final_conv = nn.Conv2d(num_internal_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True) # Bias can be true for final layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects x to be (N, C, H, W)
        if x.ndim == 3: # If (N,H,W) or (C,H,W) and C=1, add channel dim
            if x.shape[0] == 1 and self.initial_conv.in_channels == 1: # (1,H,W)
                 x = x.unsqueeze(0) # (1,1,H,W) if batch size was 1. Or it's (C,H,W)
            elif x.ndim == 3 and self.initial_conv.in_channels == x.shape[0]: # (C,H,W)
                x = x.unsqueeze(0) # (1,C,H,W)
            elif x.ndim == 3 and self.initial_conv.in_channels == 1: # (N,H,W) assumed
                 x = x.unsqueeze(1) # (N,1,H,W)
            else:
                raise ValueError(f"Input tensor shape {x.shape} not directly compatible with in_channels {self.initial_conv.in_channels}")
        elif x.ndim == 2: # (H,W)
            x = x.unsqueeze(0).unsqueeze(0) # (1,1,H,W)
            if self.initial_conv.in_channels != 1:
                 raise ValueError(f"Input tensor shape {x.shape} (from H,W) not compatible with in_channels {self.initial_conv.in_channels}")


        out = self.initial_conv(x)
        out = self.bn_initial(out)
        out = self.relu_initial(out)
        
        out = self.resnet_blocks(out)
        
        out = self.final_conv(out)
        
        return out

if __name__ == '__main__':
    # Example Usage (simple test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For a single complex image (real and imag as channels)
    denoiser_complex = SimpleResNetDenoiser(in_channels=2, out_channels=2, num_internal_channels=32, num_blocks=3).to(device)
    dummy_complex_image = torch.randn(1, 2, 64, 64, device=device) # Batch=1, Channels=2 (real/imag), H=64, W=64
    denoised_complex_image = denoiser_complex(dummy_complex_image)
    print(f"Complex Denoiser: Input shape {dummy_complex_image.shape}, Output shape {denoised_complex_image.shape}")

    # For a single magnitude image
    denoiser_mag = SimpleResNetDenoiser(in_channels=1, out_channels=1, num_internal_channels=32, num_blocks=3).to(device)
    dummy_mag_image = torch.randn(1, 1, 64, 64, device=device) # Batch=1, Channels=1, H=64, W=64
    denoised_mag_image = denoiser_mag(dummy_mag_image)
    print(f"Magnitude Denoiser: Input shape {dummy_mag_image.shape}, Output shape {denoised_mag_image.shape}")

    # Test with (H,W) input for magnitude
    dummy_hw_image = torch.randn(64, 64, device=device)
    denoised_hw_image = denoiser_mag(dummy_hw_image)
    print(f"Magnitude Denoiser with (H,W) input: Input shape {dummy_hw_image.shape}, Output shape {denoised_hw_image.shape}")
