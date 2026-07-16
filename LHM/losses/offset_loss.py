import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ACAP_Loss", "Heuristic_ACAP_Loss"]


class ACAP_Loss(nn.Module):
    """As close as possibel loss"""

    def forward(self, offset, d=0.005, **params):
        """
        ACAP (As Close As Possible) Loss.
        Encourages offset vectors to have small magnitude.

        Data Analysis (iter 8-16):
        - offset norm range: 0.002 ~ 0.032
        - offset norm mean: 0.006 ~ 0.020
        - d=0.005 provides optimal balance between regularization strength
          and loss smoothness

        Args:
            offset: tensor of shape [..., 3] representing 3D offset vectors
            d: threshold below which no penalty is applied (default: 0.005)

        Returns:
            mean loss value
        """

        offset_norm = offset.norm(p=2, dim=-1)

        offset_loss = torch.clamp(offset_norm, min=d) - d

        return offset_loss.mean()

class Heuristic_ACAP_Loss(nn.Module):
    """As close as possibel loss"""

    def __init__(self):
        super(Heuristic_ACAP_Loss, self).__init__()

    def _heurisitic_loss(self, _offset_loss):
        _loss = 0.0
        for key in self.group_dict.keys():
            key_weights = self.group_dict[key]
            group_mapping_idx = self.group_body_mapping[key]
            _loss += key_weights * _offset_loss[:, group_mapping_idx].mean()

        return _loss

    def forward(self, offset, d=0.05625, **params):
        """Empirically, where d is the thresold of distance points leave from human prior model, 1.8/32 = 0.0562."""
        "human motion or rotation is very different in each body parts, for example, the head is more stable than the leg and hand, so we use heuristic_ball_loss"

        _offset_loss = torch.clamp(offset.norm(p=2, dim=-1), min=d) - d

        return self._heurisitic_loss(_offset_loss)
