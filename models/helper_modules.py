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
            # input mean:
            x_mean = torch.mean(x, dim=0, keepdim=True)
            new_mean = (1 - g) * self.mean + g * x_mean
            # estimate of input variance:
            x_var = torch.mean((x - new_mean) ** 2, dim=0, keepdim=True)
            new_std = ((1 - g) * self.std ** 2 + g * x_var) ** 0.5
            self.std.data = new_std
            self.mean.data = new_mean
        x = (x - self.mean) / (self.std + self.eps)
        return x
