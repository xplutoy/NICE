import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *


# copy from https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


fm = 64
layers = [
    MaskedConv2d('A', 1, fm, 7, 1, 3, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True)
]
for l in range(7):
    layers.extend([
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False),
        nn.BatchNorm2d(fm),
        nn.ReLU(True)
    ])
layers.append(nn.Conv2d(fm, 256, 1))
net = nn.Sequential(*layers).to(DEVICE)
print_network(net)

trainer = optim.Adam(net.parameters())
criterion = torch.nn.CrossEntropyLoss()

lr = 1e-3
n_epochs = 30

for epoch in range(n_epochs):
    net.train()

    for i, (x, _) in enumerate(train_iter):
        x = x.to(DEVICE)
        t = (x * 255).squeeze().long()
        loss = criterion(net(x), t)

        trainer.zero_grad()
        loss.backward()
        trainer.step()

        if i % 100 == 0:
            print('[%2d/%2d] %3d loss: %.3f' % (epoch + 1, n_epochs, i + 1, loss.item()))

    with torch.no_grad():
        net.eval()
        total_loss = 0
        for i, (x, _) in enumerate(test_iter):
            x = x.to(DEVICE)
            t = (x * 255).squeeze().long()
            loss = criterion(net(x), t)
            total_loss += loss.item()
        loss = total_loss / len(test_iter)
        print('VALID_loss: %.3f' % loss)

        sample = torch.zeros(64, 1, 28, 28, device=DEVICE)

        for i in range(28):
            for j in range(28):
                out = net(sample)
                probs = F.softmax(out[:, :, i, j])
                sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
        tv.utils.save_image(sample, SAVE_DIR + 'simple_pixelcnn_{}.png'.format(epoch + 1))
