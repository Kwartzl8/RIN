import torch
import torch.nn as nn

class BregmanDivergence(nn.Module):
    def forward(self, truth, estimate):
        raise NotImplementedError("Bregman Divergence is not implemented yet. Please implement the forward method.")
    
class L2Loss(BregmanDivergence):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, truth, estimate):
        return self.loss(truth, estimate)
    
class ExponentialLoss(BregmanDivergence):
    def __init__(self):
        super(ExponentialLoss, self).__init__()

    def forward(self, truth, estimate):
        # Needs checking
        return torch.mean(torch.expm1(truth - estimate) - (truth - estimate))