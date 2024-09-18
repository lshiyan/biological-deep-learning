import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, shape, eps=1e-5, momentum=0.1, device=None):
        super(BatchNorm, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.eps = eps
        self.momentum = momentum
        self.shape = shape
        self.mean = nn.Parameter(torch.zeros(shape, device=device).unsqueeze(0), requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape, device=device).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        if self.training:
            g = self.momentum
            std, mean = torch.std_mean(x, dim=0, keepdim=True, unbiased=True)
            new_mean = (1 - g) * self.mean + g * mean
            new_std = ((1 - g) * self.std ** 2 + g * std ** 2) ** 0.5
            self.std = nn.Parameter(new_std)
            self.mean = nn.Parameter(new_mean)
        x = (x - self.mean) / (self.std + self.eps)
        return x
