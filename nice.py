import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

from utils import *

EPS = 1e-10


class additive_coupling_layer(nn.Module):
    def __init__(self, in_features, hidden_dim, permutation='reverse'):
        """
        由于pytorch不能很好的支持高级索引，该模块暂时不能使用
        :param in_features:
        :param hidden_dim:
        :param permutation: shuffle or reverse
        """
        super(additive_coupling_layer, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.permutation = np.arange(in_features)[::-1]
        if permutation == 'shuffle':
            np.random.shuffle(self.permutation)

        assert in_features % 2 == 0
        self.split = in_features // 2
        self.m = nn.Sequential(
            nn.Linear(self.split, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.split)
        )

    def forward(self, x, inv=False):
        if not inv:
            right = x[:, self.split:]
            left = self.m(x[:, self.split:]) + x[:, :self.split]
            return torch.cat([left, right], 1)[:, torch.tensor(self.permutation, dtype=torch.long)]
        else:
            rx = x[:, torch.tensor(np.argsort(self.permutation), dtype=torch.long)]
            right = rx[:, self.split:]
            left = rx[:, :self.split] - self.m(rx[:, self.split:])
            return torch.cat([left, right], 1)


class additive_alternant_coupling_layer(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(additive_alternant_coupling_layer, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        assert in_features % 2 == 0
        self.split = in_features // 2
        self.m1 = nn.Sequential(
            nn.Linear(self.split, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.split)
        )
        self.m2 = nn.Sequential(
            nn.Linear(self.split, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.split)
        )

    def forward(self, x, inv=False):
        if not inv:
            x0 = x[:, :self.split]
            x1 = x[:, self.split:]
            h1 = self.m1(x0) + x1
            h0 = x0
            y0 = h0 + self.m2(h1)
            y1 = h1
            return torch.cat([y0, y1], 1)
        else:
            y0 = x[:, :self.split]
            y1 = x[:, self.split:]
            h1 = y1
            h0 = y0 - self.m2(y1)
            x0 = h0
            x1 = h1 - self.m1(h0)
            return torch.cat([x0, x1], 1)


class nice(nn.Module):
    def __init__(self, im_size):
        super(nice, self).__init__()
        self.cp1 = additive_alternant_coupling_layer(im_size, 1024)
        self.cp2 = additive_alternant_coupling_layer(im_size, 1024)
        self.cp3 = additive_alternant_coupling_layer(im_size, 1024)
        self.scale = Parameter(torch.zeros(im_size))
        self.to(DEVICE)

    def forward(self, x, inv=False):
        if not inv:
            cp1 = self.cp1(x)
            cp2 = self.cp2(cp1)
            cp3 = self.cp3(cp2)
            return torch.exp(self.scale) * cp3
        else:
            cp3 = x * torch.exp(-self.scale)
            cp2 = self.cp3(cp3, True)
            cp1 = self.cp2(cp2, True)
            return self.cp1(cp1, True)

    def log_logistic(self, h):
        return -(F.softplus(h) + F.softplus(-h))

    def train_loss(self, h):
        return -(self.log_logistic(h).sum(1).mean() + self.scale.sum())


# train
model = nice(784)
print_network(model)
trainer = optim.Adam(model.parameters(), lr=1e-3, betas=[0.5, 0.99])
lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=2, gamma=0.9)

n_epochs = 300

for epoch in range(n_epochs):
    model.train()
    lr_scheduler.step()
    for batch_idx, (x, _) in enumerate(train_iter):
        x = x.view(x.size(0), -1).to(DEVICE)
        h = model(x)

        loss = model.train_loss(h)
        trainer.zero_grad()
        loss.backward()
        trainer.step()

        if batch_idx % 50 == 0:
            print('[ %d / %d ] loss: %.4f' % (epoch, n_epochs, loss.item()))

    with torch.no_grad():
        model.eval()
        # 从logistics中采样
        batch_size = 64
        z = torch.rand(batch_size, 784).to(DEVICE)
        h = torch.log(z + EPS) - torch.log(1 - z)
        x = model(h, inv=True)
        tv.utils.save_image(x.view(batch_size, 1, 28, 28), SAVE_DIR + 'nice_%d.png' % epoch)
