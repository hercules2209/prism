import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from basicsr.models.archs.kb_utils import KBAFunction
from basicsr.models.archs.kb_utils import LayerNorm2d, SimpleGate


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
        '''
        #with patching
        #N = self.patch_size
        N = H//4
        assert H % N == 0 and W % N == 0, "Image size must be divisible by patch size"
        output = torch.zeros_like(x)  # Initialize output tensor
        # Loop over patches
        for i in range(0, H, N):
            for j in range(0, W, N):
                # Extract patch
                patch = x[:, :, i:i+N, j:j+N]
                fft_patch = torch.fft.fft2(patch, norm="ortho")
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
                output[:, :, i:i+N, j:j+N] = ifft_patch        
        '''
        #without patching        
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
        #ifft_patch = torch.fft.ifft2(fft_new, norm="ortho").real
        ifft_patch = torch.abs(torch.fft.ifft2(fft_new, norm="ortho"))
        output = ifft_patch
        return output
        
class KBBlock_s(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False):
        super(KBBlock_s, self).__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch    = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias=True),
            )

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.sg = SimpleGate()
        self.fca = FCAttention(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        sca = self.sca(x)
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.fca(x)
                
        return y + x * self.gamma
        
     
class KBNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=4, enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s', lightweight=True, ffn_scale=2):
        super().__init__()
        basicblock = eval(basicblock)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x