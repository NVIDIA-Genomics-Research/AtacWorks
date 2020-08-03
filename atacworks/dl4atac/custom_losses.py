#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Custom loss definitions."""

from atacworks.dl4atac.metrics import CorrCoef

import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    """Pearson loss definition."""

    def __init__(self):
        """Initialize class."""
        super(PearsonLoss, self).__init__()

    def forward(self, input, targets, eps=1e-7):
        """Calculate loss after forward propagation.

        Args:
            input: array or tensor containing values predicted by the model
            targets: array or tensor containing labels
            eps: Epsilon value to add to the loss, to prevent division by 0.

        Return:
            r_loss: Value of the Pearson correlation loss

        """
        r = CorrCoef()(input, targets, eps)
        r_loss = 1 - r
        return r_loss


class PoissonLoss(nn.Module):
    """Poisson loss definition."""

    def __init__(self):
        """Initialize class."""
        super(PoissonLoss, self).__init__()

    def forward(self, input, targets, eps=1e-7):
        """Calculate loss after forward propagation.

        Args:
            input: array or tensor containing values predicted by the model
            targets: array or tensor containing labels
            eps: Epsilon value to add to the loss, to prevent division by 0.

        Return:
            loss: Value of the Poisson loss.

        """
        loss = torch.mean(input - (targets * torch.log(input + eps)))
        return loss
