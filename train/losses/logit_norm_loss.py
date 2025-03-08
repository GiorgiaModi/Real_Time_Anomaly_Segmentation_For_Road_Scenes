import torch
import torch.nn as nn
import torch.nn.functional as F
from train.losses.cross_entropy_loss import CrossEntropyLoss2d
from train.losses.focal_loss import FocalLoss

# Code from: https://github.com/hongxin001/logitnorm_ood/blob/main/common/loss_function.py
# Paper: https://arxiv.org/pdf/2205.09310

class LogitNormLoss(nn.Module):

    def __init__(self, t=1.0, loss=None): # loss type ce or fl
        super(LogitNormLoss, self).__init__()
        self.t = t
        self.loss = loss

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t

        return self.loss(logit_norm, target)