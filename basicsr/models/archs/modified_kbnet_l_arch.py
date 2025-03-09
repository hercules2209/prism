import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from basicsr.models.archs.kb_utils import KBAFunction
from basicsr.models.archs.kb_utils import LayerNorm2d, SimpleGate

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class TransAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TransAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FCAttention(nn.Module):
    def __init__(self,channels=3,Expand_Ratio=2):
        super(FCAttention,self).__init__()        
        complex_conv_ch = int(2*channels * Expand_Ratio)
        self.gp = nn.AdaptiveAvgPool2d(1)      
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )                     

        self.complex_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels= complex_conv_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True),
                       SimpleGate(),
            nn.Conv2d(in_channels=complex_conv_ch//2, out_channels=channels*2, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True),
        )            
    def forward(self,x):
        B, C, H, W = x.shape

        fft_patch = torch.fft.fft2(x, norm="ortho")
        real, imag = fft_patch.real, fft_patch.imag
        complex_feature = torch.cat([real, imag], dim=1)
        transformed_feature = self.complex_conv(complex_feature)
        real_transformed, imag_transformed = torch.chunk(transformed_feature, 2, dim=1)
        real_attn = self.sca(self.gp(real_transformed))
        imag_attn = self.sca(self.gp(imag_transformed))
        real_new = real_transformed + real_transformed*real_attn
        imag_new = imag_transformed + imag_transformed*imag_attn
        fft_new = torch.complex(real_new, imag_new)
        ifft_patch = torch.abs(torch.fft.ifft2(fft_new, norm="ortho"))
        output = ifft_patch
        return output

class ModifiedMFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, act=True, gc=2, nset=32, k=3):
        super(ModifiedMFF, self).__init__()
        self.act = act
        self.gc = gc

        hidden_features = int(dim * ffn_expansion_factor)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # Replace original SCA with NAFNet's SCA
        # Original used dim->hidden_features, NAFNet uses same channel count throughout
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=1, padding=0, stride=1,
                      bias=True),
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

        c = hidden_features
        self.k, self.c = k, c
        self.nset = nset

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)
        interc = min(dim, 24)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )
        self.conv211 = nn.Conv2d(in_channels=dim, out_channels=self.nset, kernel_size=1)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.ga1 = nn.Parameter(torch.zeros((1, hidden_features, 1, 1)) + 1e-2, requires_grad=True)

    def forward(self, x):
        x1 = self.dwconv(x)

        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv1(x)
        x2 = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf

        if self.act:
            combined = F.gelu(x1) * x2
        else:
            combined = x1 * x2
            
        combined = combined * self.sca(combined)
        
        x = self.project_out(combined)
        return x

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)


class ModifiedKBBlock_l(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(ModifiedKBBlock_l, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        # Use the modified MFF block
        self.attn = ModifiedMFF(dim, ffn_expansion_factor, bias)
        self.ffn = FCAttention(dim, num_heads, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ModifiedKBNet_l(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=1.5, bias=False):
        super(ModifiedKBNet_l, self).__init__()
        
        # Use the modified block
        TransformerBlock = ModifiedKBBlock_l

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

def get_model_complexity_info(model, input_shape=(3, 256, 256)):
    from ptflops import get_model_complexity_info
    
    macs, params = get_model_complexity_info(
        model, input_shape, verbose=False, print_per_layer_stat=False
    )
    
    params = float(params[:-3])  
    macs = float(macs[:-4])    
    return macs, params

def print_model_complexity(model, input_shape=(3, 256, 256)):
    macs, params = get_model_complexity_info(model, input_shape)
    print(f"Model Complexity: {macs:.2f} GMac, {params:.2f} M parameters")
