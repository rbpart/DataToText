import torch.nn as nn
import torch


class CustomLoss(nn.Module):
    """
    Coverage loss
    """
    def __init__(self,regularLoss, lmbd) -> None:
        super().__init__()
        self.lmbd = lmbd
        self.criterion = regularLoss

    def forward(self, outputs, targets, attns = None):
        if attns is not None:
            src_sz, bs, tgt_sz = attns['std'].size()
            coverage_steps = attns['coverage']
            attention_steps = attns['std']
            return self.criterion(outputs,targets) + self.lmbd*(torch.minimum(attention_steps,coverage_steps)).sum()/(bs)
        else:
            return self.criterion(outputs,targets)