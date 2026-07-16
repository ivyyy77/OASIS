#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import math


class CameraEmbedder(nn.Module):
    """
    Embed camera features to a high-dimensional vector.

    Reference:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L27
    """
    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.compile
    def forward(self, x):
        return self.mlp(x)

class BodyPoseRefiner(nn.Module):
    def __init__(self,
                 total_bones=16,
                 embedding_size=48,
                 mlp_width=128,
                 mlp_depth=2,
                 **_):
        super(BodyPoseRefiner, self).__init__()

        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]

        for _ in range(0, mlp_depth):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

        self.rodriguez = RodriguesModule()

    def forward(self, pose_input):
        rvec = self.block_mlps(pose_input)
        return rvec

def xaviermultiplier(m, gain):
    """
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]\
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]\
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std

def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))

def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :,
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :,
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :,
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :,
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :,
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :,
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :,
                                                              0::2, 0::2, 0::2]

def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])

class RodriguesModule(nn.Module):
    def forward(self, rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)

            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh),
        dim=1).view(-1, 3, 3)
