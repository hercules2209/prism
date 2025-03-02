import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiseNet(nn.Module):
    def __init__(self, channels, enc_blocks=[2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], mid_blocks=2):
        super().__init__()
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Main components
        self.encoder = Encoder(channels, enc_blocks)
        self.middle_block = MiddleBlock(channels[-1], num_blocks=mid_blocks)
        self.decoder = Decoder(channels, dec_blocks)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Encoder path - collect skip connections
        encoder_features = self.encoder(x)
        
        # Middle processing
        middle_output = self.middle_block(encoder_features[-1])
        
        # Decoder path with skip connections
        decoder_features = [*encoder_features[:-1], middle_output]
        clean_image = self.decoder(decoder_features)
        
        return clean_image
        
def print_tensor_shapes(x, name="tensor"):
    if isinstance(x, (list, tuple)):
        print(f"{name} is a sequence of {len(x)} tensors:")
        for i, t in enumerate(x):
            print(f"  {name}[{i}] shape: {t.shape}")
    else:
        print(f"{name} shape: {x.shape}")

class EnhancedLocalFeatureExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pixel_attention = PixelAttention(channels)
        self.point_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        y_local = self.dw_conv(x)
        y_local_attn = self.pixel_attention(y_local)
        y_local = y_local * y_local_attn
        y_local = self.point_conv(y_local)
        return y_local

class PixelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.conv(x))

class AdaptiveChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class GatingMechanism(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        a, b = torch.chunk(x, 2, dim=1)
        return a * torch.sigmoid(b)

class FeedForwardNetwork(nn.Module):
    def __init__(self, channels, expansion_factor=2):
        super().__init__()
        hidden_channels = channels * expansion_factor
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.local_extractor = EnhancedLocalFeatureExtractor(channels)
        self.global_extractor = AdaptiveChannelAttention(channels)
        self.fusion = FeatureFusion(channels * 2, channels)
        self.gating = GatingMechanism(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.ffn = FeedForwardNetwork(channels)

    def forward(self, x):
        residual = x
        y = self.norm1(x)
        y = self.local_extractor(y)
        global_features = self.global_extractor(y)
        y = self.fusion(torch.cat([y, global_features], dim=1))
        y = self.gating(y)
        y = residual + y  # First residual connection
        
        residual = y
        y = self.norm2(y)
        y = self.ffn(y)
        return residual + y  # Second residual connection

class Encoder(nn.Module):
    def __init__(self, channels, enc_blocks):
        super().__init__()
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        # Create encoder stages with multiple blocks per stage
        for i in range(len(channels)):
            # Create a sequence of encoder blocks for this stage
            blocks = nn.Sequential(*[
                EncoderBlock(channels[i]) for _ in range(enc_blocks[i])
            ])
            self.stages.append(blocks)
            
            # Add downsampling except for the last stage
            if i < len(channels) - 1:
                self.downsample.append(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=2, stride=2)
                )
    
    def forward(self, x):
        features = []
        for i, stage in enumerate(self.stages):
            # Process through all blocks in this stage
            x = stage(x)
            features.append(x)
            
            # Downsample if not the last stage
            if i < len(self.stages) - 1:
                x = self.downsample[i](x)
        return features

class MultiscaleFeatureExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=4, dilation=4)
        self.branch4 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=8, dilation=8)
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return self.fusion(torch.cat([y1, y2, y3, y4], dim=1))

class MiddleBlock(nn.Module):
    def __init__(self, channels, num_blocks=2):
        super().__init__()
        # Create a sequence of middle blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.BatchNorm2d(channels),
                'local_extractor': EnhancedLocalFeatureExtractor(channels),
                'global_extractor': AdaptiveChannelAttention(channels),
                'multiscale_extractor': MultiscaleFeatureExtractor(channels),
                'fusion': FeatureFusion(channels * 3, channels),
                'gating': GatingMechanism(channels),
                'norm2': nn.BatchNorm2d(channels),
                'ffn': FeedForwardNetwork(channels, expansion_factor=4)
            }))

    def forward(self, x):
        for block in self.blocks:
            residual = x
            y = block['norm1'](x)
            local_features = block['local_extractor'](y)
            global_features = block['global_extractor'](y)
            multiscale_features = block['multiscale_extractor'](y)
            y = block['fusion'](torch.cat([local_features, global_features, multiscale_features], dim=1))
            y = block['gating'](y)
            y = residual + y  # First residual connection
            
            residual = y
            y = block['norm2'](y)
            y = block['ffn'](y)
            x = residual + y  # Second residual connection
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Process skip connection to match desired output channels
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),  # in_channels//2 for skip connection
            nn.BatchNorm2d(out_channels)
        )
        
        # Upsampling branch
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.PixelShuffle(2)  # This will reduce channels by 4 and increase spatial dims by 2
        )
        
        # After concatenation we'll have out_channels * 2 (skip + upsampled)
        merge_channels = out_channels * 2
        
        # Feature processing
        self.local_extractor = EnhancedLocalFeatureExtractor(merge_channels)
        self.global_extractor = AdaptiveChannelAttention(merge_channels)
        
        # Final fusion and refinement
        self.fusion = nn.Sequential(
            nn.Conv2d(merge_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # Process skip connection first
        skip_processed = self.skip_conv(skip)
        
        # Upsample main features
        x_up = self.upsample_conv(x)
        
        # Verify spatial dimensions match
        if x_up.shape[2:] != skip_processed.shape[2:]:
            skip_processed = F.interpolate(
                skip_processed, 
                size=x_up.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate features
        combined = torch.cat([x_up, skip_processed], dim=1)
        
        # Process features
        local_features = self.local_extractor(combined)
        global_features = self.global_extractor(combined)
        
        # Combine and refine
        features = torch.cat([local_features, global_features], dim=1)
        out = self.fusion(features)
        out = self.final_conv(out)
        
        return out

class DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.initial_block = DecoderBlock(in_channels, out_channels)
        
        # Additional processing blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks - 1):  # -1 because we already have the initial block
            self.blocks.append(EncoderBlock(out_channels))  # Reusing EncoderBlock for consistency
    
    def forward(self, x, skip):
        # Initial decoder block with skip connection
        x = self.initial_block(x, skip)
        
        # Additional processing blocks
        for block in self.blocks:
            x = block(x)
            
        return x

class Decoder(nn.Module):
    def __init__(self, channels, dec_blocks):
        super().__init__()
        # channels is [64, 128, 256, 512]
        channels_rev = channels[::-1]  # [512, 256, 128, 64]
        dec_blocks_rev = dec_blocks[::-1]  # Reverse to match channels
        
        # Create decoder stages
        self.stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch = channels_rev[i]
            out_ch = channels_rev[i + 1]
            self.stages.append(DecoderStage(in_ch, out_ch, dec_blocks_rev[i]))
        
        # Final output convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # Split features into encoder outputs and middle features
        encoder_features = features[:-1]  # All except last
        x = features[-1]  # Middle output
        
        # Reverse encoder features for skip connections
        encoder_features = encoder_features[::-1]
        
        # Process through decoder stages
        for i, stage in enumerate(self.stages):
            skip = encoder_features[i]
            x = stage(x, skip)
        
        # Final convolution
        output = self.final_conv(x)
        
        return output
