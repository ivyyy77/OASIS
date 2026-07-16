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

__all__ = ['LPIPSLoss']


class LPIPSLoss(nn.Module):
    """
    Compute LPIPS loss between two images.
    """

    def __init__(self, device, prefech: bool = False):
        super().__init__()
        self.device = device
        self.cached_models = {}
        self.loss_fn = self._get_model('vgg')
        if prefech:
            self.prefetch_models()

    def _get_model(self, model_name: str):
        if model_name not in self.cached_models:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                import lpips
                _model = lpips.LPIPS(net=model_name, eval_mode=True, verbose=False).to(self.device)
            _model = torch.compile(_model)
            self.cached_models[model_name] = _model
        return self.cached_models[model_name]

    def prefetch_models(self):
        _model_names = ['alex', 'vgg']
        for model_name in _model_names:
            self._get_model(model_name)

    def forward(self, x, y, is_training: bool = True):
        """
        Assume images are 0-1 scaled and channel first.

        Args:
            x: [N, M, C, H, W]
            y: [N, M, C, H, W]
            is_training: whether to use VGG or AlexNet.

        Returns:
            Mean-reduced LPIPS loss across batch.
        """

        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        y = y.reshape(N*M, C, H, W)
        image_loss = self.loss_fn(x, y, normalize=True).mean(dim=[1, 2, 3])
        batch_loss = image_loss.reshape(N, M).mean(dim=1)
        all_loss = batch_loss.mean()
        return all_loss

    def test(self, x, y, is_training: bool = True):
        """
        Args:

        Returns:
        """

        if x.ndim == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if x.ndim == 5:
            N, M, C, H, W = x.shape
            x = x.reshape(N*M, C, H, W)
            y = y.reshape(N*M, C, H, W)
        elif x.ndim == 4:
            N, C, H, W = x.shape
            M = 1
        else:
            raise ValueError(f"Unsupported shape {x.shape}")

        image_loss = self.loss_fn(x, y, normalize=True)
        image_loss = image_loss.view(-1)

        if M > 1:
            batch_loss = image_loss.view(N, M).mean(dim=1)
        else:
            batch_loss = image_loss

        all_loss = batch_loss.mean()
        return all_loss
