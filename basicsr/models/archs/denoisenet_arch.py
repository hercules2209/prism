"""
DenoiseNet architecture registration for BasicSR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class DenoiseNet(nn.Module):
    """Registered DenoiseNet architecture"""
    def __init__(self, channels):
        super().__init__()
        self.encoder = Encoder(channels)
        self.middle_block = MiddleBlock(channels[-1])
        self.decoder = Decoder(channels[::-1])
        
    def forward(self, x):
        encoder_features = self.encoder(x)
        middle_output = self.middle_block(encoder_features[-1])
        clean_image = self.decoder([*encoder_features[:-1], middle_output])
        return clean_image

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
        self.norm1 = nn.LayerNorm(channels)
        self.local_extractor = EnhancedLocalFeatureExtractor(channels)
        self.global_extractor = AdaptiveChannelAttention(channels)
        self.fusion = FeatureFusion(channels * 2, channels)
        self.gating = GatingMechanism(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FeedForwardNetwork(channels)
        
    def forward(self, x):
        y = self.norm1(x)
        y_local = self.local_extractor(y)
        y_global = self.global_extractor(y)
        y_fused = self.fusion(torch.cat([y_local, y_global], dim=1))
        y_gated = self.gating(y_fused)
        y = x + y_gated
        return y + self.ffn(self.norm2(y))

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
        self.norm1 = nn.LayerNorm(channels)
        self.local_extractor = EnhancedLocalFeatureExtractor(channels)
        self.global_extractor = AdaptiveChannelAttention(channels)
        self.multiscale_extractor = MultiscaleFeatureExtractor(channels)
        self.fusion = FeatureFusion(channels * 3, channels)
        self.gating = GatingMechanism(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FeedForwardNetwork(channels, expansion_factor=4)

    def forward(self, x):
        y = self.norm1(x)
        y_local = self.local_extractor(y)
        y_global = self.global_extractor(y)
        y_multiscale = self.multiscale_extractor(y)
        y_fused = self.fusion(torch.cat([y_local, y_global, y_multiscale], dim=1))
        y_gated = self.gating(y_fused)
        y = x + y_gated
        return y + self.ffn(self.norm2(y))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.upsample = nn.PixelShuffle(2)
        self.conv_up = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        
        self.skip_fusion = FeatureFusion(out_channels + skip_channels, out_channels)
        
        self.local_extractor = EnhancedLocalFeatureExtractor(out_channels)
        self.global_extractor = AdaptiveChannelAttention(out_channels)
        self.fusion = FeatureFusion(out_channels * 2, out_channels)
        self.gating = GatingMechanism(out_channels)
        
        self.norm = nn.LayerNorm(out_channels)
        self.ffn = FeedForwardNetwork(out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.conv_up(x)
        x = self.skip_fusion(torch.cat([x, skip], dim=1))
        
        y_local = self.local_extractor(x)
        y_global = self.global_extractor(x)
        y_fused = self.fusion(torch.cat([y_local, y_global], dim=1))
        y_gated = self.gating(y_fused)
        
        y = x + y_gated
        return y + self.ffn(self.norm(y))

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i+1], channels[-(i+2)])
            for i in range(len(channels) - 1)
        ])
        self.final_conv = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, features):
        x = features[-1]
        for i, block in enumerate(self.blocks):
            x = block(x, features[-(i+2)])
        return self.activation(self.final_conv(x))

class DenoiseNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = Encoder(channels)
        self.middle_block = MiddleBlock(channels[-1])
        self.decoder = Decoder(channels[::-1])
        
    def forward(self, x):
        encoder_features = self.encoder(x)
        middle_output = self.middle_block(encoder_features[-1])
        clean_image = self.decoder([*encoder_features[:-1], middle_output])
        return clean_image


