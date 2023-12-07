import torch
import torch.nn as nn
import torch.optim as optim


class MINE(nn.Module):
    def __init__(self, input_dim):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1)
        self.activation = nn.ReLU()

    def forward(self, x, y):
        joint_xy = torch.cat([x, y], dim=1)

        # Instead of repeating, we shuffle y to get samples from the marginal distribution of Y
        shuffled_y = y[torch.randperm(y.size(0))]
        marginal_xy = torch.cat([x, shuffled_y], dim=1)

        t_xy = self.fc2(self.activation(self.fc1(joint_xy)))
        t_x_y = self.fc2(self.activation(self.fc1(marginal_xy)))

        mi_loss = (torch.mean(t_xy) - torch.log(torch.mean(torch.exp(t_x_y)))).mean()
        return -mi_loss


models = MINE(64)


def cacul_mine(x, y, idx):
    optimizer = optim.Adam(models.parameters(), lr=0.0001)

    optimizer.zero_grad()
    loss = models(x, y)
    loss.backward()
    optimizer.step()

    if idx % 100 == 0:
        print(f"Epoch {idx}, MI estimation: {-loss.item()}")
