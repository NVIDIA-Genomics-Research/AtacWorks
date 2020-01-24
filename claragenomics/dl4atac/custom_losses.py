#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import torch
import torch.nn as nn
from claragenomics.dl4atac.metrics import CorrCoef

class PearsonLoss(nn.Module):

    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, input, targets, eps=1e-7):
        r = CorrCoef()(input, targets, eps)
        r_loss = 1 - r
        return r_loss


class PoissonLoss(nn.Module):

    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, input, targets, eps=1e-7):
        loss = torch.mean(input - (targets*torch.log(input + eps)))
        return loss
