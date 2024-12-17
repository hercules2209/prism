"""
DenoiseNet architecture registration for BasicSR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiseNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Main components
        self.encoder = Encoder(channels)
        self.middle_block = MiddleBlock(channels[-1])
        self.decoder = Decoder(channels)
    
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
        y = self.norm1(x)
        y = self.local_extractor(y)
        global_features = self.global_extractor(y)
        y = self.fusion(torch.cat([y, global_features], dim=1))
        y = self.gating(y)
        y = self.norm2(y)
        y = self.ffn(y)
        return y

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(channels[i]) for i in range(len(channels))
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i+1], kernel_size=2, stride=2)
            for i in range(len(channels)-1)
        ])
    
    def forward(self, x):
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if i < len(self.blocks) - 1:
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
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.local_extractor = EnhancedLocalFeatureExtractor(channels)
        self.global_extractor = AdaptiveChannelAttention(channels)
        self.multiscale_extractor = MultiscaleFeatureExtractor(channels)
        self.fusion = FeatureFusion(channels * 3, channels)
        self.gating = GatingMechanism(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.ffn = FeedForwardNetwork(channels, expansion_factor=4)

    def forward(self, x):
        y = self.norm1(x)
        local_features = self.local_extractor(y)
        global_features = self.global_extractor(y)
        multiscale_features = self.multiscale_extractor(y)
        y = self.fusion(torch.cat([local_features, global_features, multiscale_features], dim=1))
        y = self.gating(y)
        y = self.norm2(y)
        y = self.ffn(y)
        return y

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

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # channels is [64, 128, 256, 512]
        channels_rev = channels[::-1]  # [512, 256, 128, 64]
        
        # Create decoder blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch = channels_rev[i]
            out_ch = channels_rev[i + 1] if i < len(channels_rev) - 1 else 64
            self.blocks.append(DecoderBlock(in_ch, out_ch))
        
        # Final output convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
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
        
        # Process through decoder blocks
        for i, block in enumerate(self.blocks):
            skip = encoder_features[i]
            x = block(x, skip)
        
        # Final convolution
        output = self.final_conv(x)
        
        return output