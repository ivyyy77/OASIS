
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

class SimpleUNet(nn.Module):
    def __init__(
        self, in_size, out_size, in_dim, out_dim,
        channel_scale=1,
    ): 
        super().__init__()
        self.in_size, self.out_size = in_size, out_size
        channels = {
            '4': 256, '8': 256, '16': 256, '32': 256,
            '64': 128, '128': 64, '256': 32, '512': 16, '1024': 8
        }
        for key in channels.keys():
            channels[key] = int(channels[key] / channel_scale)

        assert in_size <= out_size * 2, f'In/out: {in_size}/{out_size}.'
        assert f'{in_size}' in channels.keys(), f'In size: {in_size}.'
        assert f'{out_size}' in channels.keys(), f'Out size: {out_size}.'
        self.log_size = int(math.log(out_size, 2))

        ### UNet Module
        if self.in_size <= self.out_size:
            self.conv_body_first = nn.Conv2d(in_dim, channels[f'{out_size}'], 1)
        else:
            self.conv_body_first = nn.ModuleList([
                nn.Conv2d(in_dim, channels[f'{in_size}'], 1),
                ResBlock(channels[f'{in_size}'], channels[f'{out_size}'], mode='down'),
            ])

        # Downsample
        in_channels = channels[f'{out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)

        # Upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels
            
        # To RGB
        self.toRGB = nn.Conv2d(channels[f'{2**self.log_size}'], out_dim, 1)

    def forward(self, x):
        unet_skips = []
        # Resize input if smaller than target size
        if x.shape[-1] < self.out_size:
            x = nn.functional.interpolate(
                x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False
            )
        # UNet downsample
        if self.in_size <= self.out_size:
            feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        else:
            feat = F.leaky_relu_(self.conv_body_first[0](x), negative_slope=0.2)
            feat = self.conv_body_first[1](feat)

        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)

        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        # UNet upsample
        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)

        # Generate final RGB image
        image = self.toRGB(feat)
        return image

class StyleUNet(nn.Module):
    def __init__(
        self, in_size, out_size, in_dim, out_dim, 
        num_style_feat=512, num_mlp=8, activation=True,
        channel_scale=1,small=False,extra_style_dim=-1,
    ): 
        super().__init__()
        
        self.activation = activation
        self.num_style_feat = num_style_feat
        self.in_size, self.out_size = in_size, out_size
        channels = {
            '4': 256, '8': 256, '16': 256, '32': 256,
            '64': 128, '128': 64, '256': 32, '512': 16, '1024': 8
        }
        for key in channels.keys():
            channels[key]=int(channels[key]/channel_scale)
        
        assert in_size <= out_size*2, f'In/out: {in_size}/{out_size}.'
        assert f'{in_size}' in channels.keys(), f'In size: {in_size}.'
        assert f'{out_size}' in channels.keys(), f'Out size: {out_size}.'
        self.log_size = int(math.log(out_size, 2))
        ### UNET Module
        if self.in_size <= self.out_size:
            self.conv_body_first = nn.Conv2d(in_dim, channels[f'{out_size}'], 1)
        else:
            self.conv_body_first = nn.ModuleList([
                nn.Conv2d(in_dim, channels[f'{in_size}'], 1),
                ResBlock(channels[f'{in_size}'], channels[f'{out_size}'], mode='down'),
            ])
        # downsample
        in_channels = channels[f'{out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)
        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels

        ### STYLE Module
        # condition
        self.final_linear = nn.Linear(channels['4'] * 4 * 4, num_style_feat)
        self.extra_style_dim=extra_style_dim
        if extra_style_dim>0:
            self.style_fuse=nn.Sequential(nn.Linear(extra_style_dim+num_style_feat,num_style_feat),
                                          nn.LeakyReLU(0.2, True),
                                          nn.Linear(num_style_feat,num_style_feat),)
        if small:
            self.stylegan_decoder = StyleGAN2GeneratorCSFT_small(
                out_dim=out_dim, out_size=out_size, 
                num_style_feat=num_style_feat, num_mlp=num_mlp,
                channel_scale=channel_scale,
            )
        else:
            self.stylegan_decoder = StyleGAN2GeneratorCSFT(
                out_dim=out_dim, out_size=out_size, 
                num_style_feat=num_style_feat, num_mlp=num_mlp,
                channel_scale=channel_scale,
            )
        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            ch = channels[f'{2**i}']
            self.condition_scale.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch, ch, 3, 1, 1)#* 2
            ))
            self.condition_shift.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch, ch, 3, 1, 1)#x* 2
            ))

    def forward(self, x, randomize_noise =True,extra_style=None):
        conditions, unet_skips, out_rgbs = [], [], []
        # size
        
        if x.shape[-1] < self.out_size:
            x = nn.functional.interpolate(
                x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False
            )
        # UNET downsample
        if self.in_size <= self.out_size:
            feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        else:
            feat = F.leaky_relu_(self.conv_body_first[0](x), negative_slope=0.2)
            feat = self.conv_body_first[1](feat)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)
        # style code
        style_code = self.final_linear(feat.reshape(feat.size(0), -1))
        if self.extra_style_dim>0 and extra_style is not None:
            style_code=self.style_fuse(torch.cat([style_code,extra_style],dim=1))
        # UNET upsample
        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)
            # SFT module
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())

        image = self.stylegan_decoder(
            style_code, conditions, randomize_noise=randomize_noise
        )
        # activation
        if self.activation:
            image = torch.sigmoid(image)
           
        return image
    


class StyleGAN2GeneratorCSFT(nn.Module):
    # StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).
    def __init__(
            self, out_size, out_dim=3, num_style_feat=512, num_mlp=8,channel_scale=1,
        ):
        super().__init__()
        channels = {
            '4': 256, '8': 256, '16': 256, '32': 256,
            '64': 128, '128': 64, '256': 32, '512': 16, '1024': 8
        }
        for key in channels.keys():
            channels[key]=int(channels[key]/channel_scale)
        self.channels = channels
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.extend([
                nn.Linear(num_style_feat, num_style_feat, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        # initialization
        default_init_weights(self.style_mlp, scale=1, bias_fill=0, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        # Upsample First layer
        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'], channels['4'], kernel_size=3,
            num_style_feat=num_style_feat, demodulate=True, sample_mode=None
        )
        self.to_rgb1 = ToRGB(channels['4'], out_dim, num_style_feat, upsample=False)
        # Upsample 
        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        in_channels = channels['4']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels, out_channels, kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True, sample_mode='upsample'
                )
            )
            self.style_convs.append(
                StyleConv(
                    out_channels, out_channels, kernel_size=3,
                    num_style_feat=num_style_feat, 
                    demodulate=True, sample_mode=None
                )
            )
            self.to_rgbs.append(
                ToRGB(out_channels, out_dim, num_style_feat, upsample=True)
            )
            in_channels = out_channels

    def forward(self, styles, conditions, randomize_noise=True):
        # Forward function for StyleGAN2GeneratorCSFT.
        styles = self.style_mlp(styles)
        # noises
        if randomize_noise:
            noise = [None] * self.num_layers
        else:
            noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # get style latents with injection
        inject_index = self.num_latent
        # repeat latent code for all the layers
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles
        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.style_convs[::2], self.style_convs[1::2], 
                noise[1::2], noise[2::2], self.to_rgbs
            ):
            out = conv1(out, latent[:, i], noise=noise1)
            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                out = out * conditions[i - 1] + conditions[i]
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2
        image = skip
        return image

class StyleGAN2GeneratorCSFT_small(nn.Module):
    # StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).
    def __init__(
            self, out_size, out_dim=3, num_style_feat=512, num_mlp=8,channel_scale=1,
        ):
        super().__init__()
        channels = {
            '4': 256, '8': 256, '16': 256, '32': 256,
            '64': 128, '128': 64, '256': 32, '512': 16, '1024': 8
        }
        for key in channels.keys():
            channels[key]=int(channels[key]/channel_scale)
        self.channels = channels
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 1 + 1
        self.num_latent = self.log_size * 1 
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.extend([
                nn.Linear(num_style_feat, num_style_feat, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        # initialization
        default_init_weights(self.style_mlp, scale=1, bias_fill=0, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        # Upsample First layer
        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'], channels['4'], kernel_size=3,
            num_style_feat=num_style_feat, demodulate=True, sample_mode=None
        )
        self.to_rgb1 = ToRGB(channels['4'], out_dim, num_style_feat, upsample=False)
        # Upsample 
        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        self.normal_convs = nn.ModuleList()
        in_channels = channels['4']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels, out_channels, kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True, sample_mode='upsample'
                )
            )
            self.normal_convs.append(nn.Sequential(nn.Conv2d(out_channels,out_channels, kernel_size=3,padding=1,),nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            self.to_rgbs.append(
                ToRGB(out_channels, out_dim, num_style_feat, upsample=True)
            )
            in_channels = out_channels

    def forward(self, styles, conditions, randomize_noise=True):
        # Forward function for StyleGAN2GeneratorCSFT.
        styles = self.style_mlp(styles)
        # noises
        if randomize_noise:
            noise = [None] * self.num_layers
        else:
            noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # get style latents with injection
        inject_index = self.num_latent
        # repeat latent code for all the layers
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles
        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        
        for conv1,conv2, noise1, to_rgb in zip(
                self.style_convs, self.normal_convs,
                noise[1:], self.to_rgbs
            ):
            out = conv1(out, latent[:, i], noise=noise1)
            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                out = out * conditions[(i-1)*2] + conditions[(i-1)*2+1]
            out=conv2(out)
            skip = to_rgb(out, latent[:, i+1], skip)  # feature back to the rgb space
            i += 1
        image = skip
        return image


class ResBlock(nn.Module):
    """
    Residual block with bilinear upsampling/downsampling.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        mode (str): Upsampling/downsampling mode. Options: down | up. Default: down.
    """
    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        # upsample/downsample
        out = F.interpolate(
            out, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        # skip
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )
        skip = self.skip(x)
        out = out + skip
        return out
    
class NormStyleCode(nn.Module):
    def forward(self, x):
        # Normalize the style codes.
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
    

class ConstantInput(nn.Module):
    def __init__(self, num_channel, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out
    

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """
    Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
                    
class StyleConv(nn.Module):
    # Style conv used in StyleGAN2.
    def __init__(
            self, in_channels, out_channels, kernel_size, num_style_feat, 
            demodulate=True, sample_mode=None
        ):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, num_style_feat, 
            demodulate=demodulate, sample_mode=sample_mode
        )
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style, noise=None):
        # modulate
        out = self.modulated_conv(x, style) * 2 ** 0.5  # for conversion
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        # add bias
        out = out + self.bias
        # activation
        out = self.activate(out)
        return out
    
class ModulatedConv2d(nn.Module):
    # Modulated Conv2d used in StyleGAN2. (No bias in ModulatedConv2d.)
    def __init__(
            self, in_channels, out_channels, kernel_size, num_style_feat, 
            demodulate=True, sample_mode=None, eps=1e-8
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        # modulation inside each modulated conv
        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        # initialization
        default_init_weights(self.modulation, scale=1, bias_fill=1, a=0, mode='fan_in', nonlinearity='linear')
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size) /
            math.sqrt(in_channels * kernel_size**2)
        )
        self.padding = kernel_size // 2

    def forward(self, x, style):
        b, c, h, w = x.shape
        # weight modulation
        style = self.modulation(style).view(b, 1, c, 1, 1)
        # self.weight: (1, c_out, c_in, k, k); style: (b, 1, c, 1, 1)
        weight = self.weight * style  # (b, c_out, c_in, k, k)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)
        # upsample or downsample if necessary
        if self.sample_mode == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif self.sample_mode == 'downsample':
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        # weight: (b*c_out, c_in, k, k), groups=b
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, *out.shape[2:4])
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, demodulate={self.demodulate}, sample_mode={self.sample_mode})'
        )

class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, num_style_feat, upsample=True):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size=1, 
            num_style_feat=num_style_feat, demodulate=False, sample_mode=None
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    def forward(self, x, style, skip=None):
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return out
    

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out
    

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        bias=True,
        activate=True,
    ):
        layers = []


        stride = 2 if downsample else 1
        padding = kernel_size // 2  
        layers.append(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(0.2, inplace=True)) 
        super().__init__(*layers)
        

if __name__ == '__main__':
    import time
    from tqdm import tqdm
    with torch.no_grad():
        decode_channel=32
        iterations = 100 
        model = StyleUNet(in_size=512, in_dim=decode_channel, out_dim=3, out_size=512,num_style_feat=512,num_mlp=8,channel_scale=1,small=True).cuda()
        model.eval()
        
        data = torch.rand(1, decode_channel, 512, 512).cuda()
        print(model(data.clone()).shape)
        
        start_time = time.time()
        for _ in tqdm(range(iterations)):
            result = model(data.clone())
        total_time = time.time() - start_time
        average_time = total_time / iterations
        fps = 1 / average_time
        print(f"StyleUNet: Decode_channel:{decode_channel}. FPS (Frames Per Second): {fps:.2f}")
        
        model = SimpleUNet(in_size=512, out_size=512, in_dim=decode_channel, out_dim=3,channel_scale=1).cuda()
        model.eval()
        print(model(data.clone()).shape)
        start_time = time.time()
        for _ in tqdm(range(iterations)):
            result = model(data.clone())
        total_time = time.time() - start_time
        average_time = total_time / iterations
        fps = 1 / average_time
        print(f"SimpleUNet: Decode_channel:{decode_channel}. FPS (Frames Per Second): {fps:.2f}")