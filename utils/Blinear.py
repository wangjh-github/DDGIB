import torch.nn as nn


class Blinear(nn.Module):
    def __init__(self, n_h):
        super(Blinear, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

    def forward(self, x, y):
        z = self.f_k(x, y)
        return z
