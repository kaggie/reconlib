import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Helper module for a double convolution block: (Conv2d -> BatchNorm2d -> ReLU) * 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

class EncoderBlock(nn.Module):
    """
    Helper module for an encoder block in the U-Net: ConvBlock -> MaxPool2d
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv_block(x)
        pooled = self.pool(skip)
        return pooled, skip

class DecoderBlock(nn.Module):
    """
    Helper module for a decoder block in the U-Net: UpConv/TransposedConv -> Concatenate -> ConvBlock
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, in_channels from the previous layer are preserved.
            # Then concatenation with skip connection happens.
            # So, ConvBlock input is in_channels (from upsampled) + in_channels // 2 (from skip connection if features halved in encoder)
            # Or more generally: in_channels (from upsampled) + skip_channels
            # Here, 'in_channels' is for the features from the layer below (that gets upsampled)
            # 'out_channels' is for the output of this decoder block's ConvBlock
            self.conv = ConvBlock(in_channels + out_channels * 2, out_channels) # if features were doubled in encoder for skip
                                                                              # A common pattern: skip_features = out_channels_current_level_encoder
                                                                              # which is typically out_channels_decoder_block * 2
                                                                              # So in_channels_conv_block = features_upsampled + features_skip
                                                                              # features_upsampled comes from in_channels (e.g. 1024)
                                                                              # features_skip comes from encoder (e.g. 512, if current decoder out is 512)
                                                                              # This implies in_channels for ConvBlock = in_channels_from_up + skip_channels
                                                                              # Let's assume skip_channels = in_channels from previous encoder level
                                                                              # if in_channels = 1024, out_channels = 512 for this decoder block
                                                                              # upsampled = 1024, skip_from_encoder = 512
                                                                              # So ConvBlock in = 1024 + 512.
                                                                              # The 'in_channels' to DecoderBlock is from previous decoder block (or bottleneck)
                                                                              # The 'out_channels' is the target for this block.
                                                                              # Skip connection will have 'out_channels' from corresponding encoder.
                                                                              # So, ConvBlock in_channels = in_channels_from_previous_decoder_upsampled + out_channels_from_encoder_skip
                                                                              # Typically, in_channels_from_previous_decoder_upsampled = in_channels
                                                                              # And out_channels_from_encoder_skip = out_channels (if symmetric U-Net)
                                                                              # So, conv_in_channels = in_channels (from upsample) + out_channels (from skip)
            self.conv = ConvBlock(in_channels + out_channels, out_channels)


        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels) # (in_channels//2 from up) + (in_channels//2 from skip) = in_channels

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        
        # Ensure dimensions match for concatenation if using padding in Conv2d
        # (padding='same' or manual padding might be needed if sizes differ due to convs without padding)
        # With kernel_size=3, padding=1, output size is same as input, so MaxPool2d is main source of change.
        # Upsample/ConvTranspose2d doubles the size.
        if x_up.shape[2:] != skip.shape[2:]:
            # If ConvTranspose2d output size is slightly different from skip connection size
            # (can happen if original image size is odd)
            # Pad x_up to match skip's H, W
            # Or use F.interpolate for upsampling, which gives more control over output_size
            # For nn.Upsample with scale_factor, it should generally align if encoder pooling was standard.
            # Let's add a check and potential crop/pad for robustness, common in U-Nets
            diff_y = skip.size()[2] - x_up.size()[2]
            diff_x = skip.size()[3] - x_up.size()[3]

            x_up = F.pad(x_up, [diff_x // 2, diff_x - diff_x // 2,
                               diff_y // 2, diff_y - diff_y // 2])

        x_concat = torch.cat([x_up, skip], dim=1)
        return self.conv(x_concat)

class UNet(nn.Module):
    """
    Standard 2D U-Net architecture.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 num_levels: int = 4,
                 initial_features: int = 64,
                 bilinear_upsampling: bool = True):
        super().__init__()
        self.num_levels = num_levels
        self.initial_features = initial_features
        self.bilinear_upsampling = bilinear_upsampling

        features = initial_features
        self.encoders = nn.ModuleList()
        for _ in range(num_levels):
            self.encoders.append(EncoderBlock(in_channels, features))
            in_channels = features
            features *= 2

        self.bottleneck = ConvBlock(in_channels, features) # features is now initial_features * 2^num_levels

        self.decoders = nn.ModuleList()
        for _ in range(num_levels):
            # Input to ConvTranspose2d or Upsample in DecoderBlock is 'features' (e.g. 1024)
            # Output channels of ConvBlock in DecoderBlock is 'features // 2' (e.g. 512)
            # Skip connection comes from encoder with 'features // 2' channels
            self.decoders.append(DecoderBlock(features, features // 2, bilinear=bilinear_upsampling))
            features //= 2
        
        # Final 1x1 convolution to map to out_channels
        self.final_conv = nn.Conv2d(initial_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        # Skips are from deepest to shallowest, so reverse the list for decoders
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(i + 1)]) # skips are [skip_level0, skip_level1, ...]

        return self.final_conv(x)

if __name__ == '__main__':
    # Test U-Net instantiation and forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example 1: Default parameters
    print("Testing U-Net with default parameters...")
    model_default = UNet().to(device)
    dummy_input_default = torch.randn(1, 1, 256, 256).to(device) # Batch_size=1, In_channels=1, H=256, W=256
    output_default = model_default(dummy_input_default)
    print(f"Input shape: {dummy_input_default.shape}")
    print(f"Output shape (default): {output_default.shape}")
    assert output_default.shape == (1, 1, 256, 256), "Default output shape mismatch!"
    print("-" * 30)

    # Example 2: Custom parameters
    print("Testing U-Net with custom parameters...")
    model_custom = UNet(in_channels=3, out_channels=2, num_levels=3, initial_features=32, bilinear_upsampling=False).to(device)
    dummy_input_custom = torch.randn(2, 3, 128, 128).to(device) # Batch_size=2, In_channels=3, H=128, W=128
    output_custom = model_custom(dummy_input_custom)
    print(f"Input shape: {dummy_input_custom.shape}")
    print(f"Output shape (custom): {output_custom.shape}")
    assert output_custom.shape == (2, 2, 128, 128), "Custom output shape mismatch!"
    print("-" * 30)

    # Example 3: Test with non-power-of-2 dimensions (U-Net should handle via padding in DecoderBlock)
    print("Testing U-Net with non-power-of-2 dimensions...")
    model_odd_dims = UNet(num_levels=2, initial_features=16).to(device)
    dummy_input_odd = torch.randn(1, 1, 96, 96).to(device) # H, W are divisible by 2^num_levels (96/4=24)
                                                       # Let's try something not easily divisible
    dummy_input_tricky = torch.randn(1, 1, 90, 90).to(device) # 90 / 4 = 22.5
    # MaxPool2d will floor: 90->45, 45->22 (num_levels=2)
    # Upsampling: 22->44, 44->88. Skip from 90 and 45.
    # So padding will be required.
    output_tricky = model_odd_dims(dummy_input_tricky)
    print(f"Input shape: {dummy_input_tricky.shape}")
    print(f"Output shape (tricky input): {output_tricky.shape}")
    assert output_tricky.shape == (1, 1, 90, 90), "Tricky input output shape mismatch!"
    print("U-Net tests completed successfully!")
    
    # You can also print model summary
    # try:
    #     from torchinfo import summary
    #     print("\nModel Summary (default):")
    #     summary(model_default, input_size=dummy_input_default.shape)
    # except ImportError:
    #     print("\n`torchinfo` not installed. Skipping model summary.")

    # print("\nModel Architecture (default):")
    # print(model_default)
