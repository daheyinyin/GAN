import os, warnings
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image



project = 'dcgan'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Z_DIM = 100
BATCH_SIZE = 32#128
LR = 0.0002
TRAIN_EPOCHS = 100
FEATURES_D = 32#64#32
IMG_CHANNELS = 1#3
IMG_SIZE = 32#32#64
fixed_z = torch.randn((5 * 5, Z_DIM, 1, 1)).to(device)    # fixed noise
real_label = 1.
fake_label = 0.

rand_path = f'result/{project}/Random_results'
fix_path = f'result/{project}/Fixed_results'
# results save folder
if not os.path.isdir(f'result/'):
    os.mkdir(f'result/')
if not os.path.isdir(f'result/{project}'):
    os.mkdir(f'result/{project}')
if not os.path.isdir(rand_path):
    os.mkdir(rand_path)
if not os.path.isdir(fix_path):
    os.mkdir(fix_path)



class Generator(nn.Module):
    def __init__(self, z_dim=64, channels_img=1, size_img = 32, features_d=64):
        super(Generator, self).__init__()
        if size_img == 64:
            self.gen = nn.Sequential(
                # input is Z, [B, 64, 1, 1] -> [B, 64 * 8, 4, 4]
                nn.ConvTranspose2d(z_dim, features_d * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features_d * 8),
                nn.ReLU(True),
                # state size. [B, 64 * 8, 4, 4] -> [B, 64 * 4, 8, 8]
                nn.ConvTranspose2d(features_d * 8, features_d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d * 4),
                nn.ReLU(True),
                # state size. [B, 64 * 4, 8, 8] -> [B, 64 * 2, 16, 16]
                nn.ConvTranspose2d(features_d * 4, features_d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d * 2),
                nn.ReLU(True),
                # state size. [B, 64 * 2, 16, 16] -> [B, 64, 32, 32]
                nn.ConvTranspose2d(features_d * 2, features_d, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d),
                nn.ReLU(True),
                # state size. [B, 64, 32, 32] -> [B, 1, 64, 64]
                nn.ConvTranspose2d(features_d, channels_img, 4, 2, 1, bias=False),
                nn.Tanh()
                # nn.Sigmoid()
            )
        elif size_img == 32:
            self.gen = nn.Sequential(
                # input is Z, [B, 64, 1, 1] -> [B, 64 * 4, 4, 4]
                nn.ConvTranspose2d(z_dim, features_d * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features_d * 4),
                nn.ReLU(True),
                # state size. [B, 64 * 4, 4, 4] -> [B, 64 * 2, 8, 8]
                nn.ConvTranspose2d(features_d * 4, features_d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d * 2),
                nn.ReLU(True),
                # state size. [B, 64 * 2, 8, 8] -> [B, 64, 16, 16]
                nn.ConvTranspose2d(features_d * 2, features_d, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d),
                nn.ReLU(True),
                # state size. [B, 64, 16, 16] -> [B, 1, 32, 32]
                nn.ConvTranspose2d(features_d, 1, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        else:
            raise Exception('size_img must be 32 or 64')
    def forward(self, input):
        return self.gen(input)

class Discriminator(nn.Module):
    def __init__(self, channels_img=1, size_img = 32, features_d=64):
        super(Discriminator, self).__init__()
        if size_img == 64:
            self.dis = nn.Sequential(
                # input [B, 1, 64, 64] -> [B, 64, 32, 32]
                nn.Conv2d(channels_img, features_d, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d),
                nn.LeakyReLU(0.1),
                # input [B, 64, 32, 32] -> [B, 128, 16, 16]
                nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d * 2),
                nn.LeakyReLU(0.1),
                # state size. [B, 128, 16, 16] -> [B, 256, 8, 8]
                nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d * 4),
                nn.LeakyReLU(0.1),
                # state size. [B, 256, 8, 8] -> [B, 512, 4, 4]
                nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d * 8),
                nn.LeakyReLU(0.1),

                # state size. [B, 512, 4, 4] -> [B, 1, 1, 1]
                nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif size_img == 32:
            self.dis = nn.Sequential(
                # input [B, 1, 32, 32] -> [B, 64, 16, 16]
                nn.Conv2d(1, features_d, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d),
                nn.LeakyReLU(0.2),
                # state size. [B, 64, 16, 16] -> [B, 128, 8, 8]
                nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d * 2),
                nn.LeakyReLU(0.2),
                # state size. [B, 128, 8, 8] -> [B, 256, 4, 4]
                nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d * 4),
                nn.LeakyReLU(0.2),
                # state size. [B, 256, 4, 4] -> [B, 1, 1, 1]
                nn.Conv2d(features_d * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        else:
            raise Exception('size_img must be 32 or 64')

    def forward(self, input):
        return self.dis(input)


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            nn.init.constant_(m.bias, 0)

def show_train_loss(hist, show = False, save = False, path = 'Train_loss.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()



transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5 for _ in range(IMG_CHANNELS)],
                         std=[0.5 for _ in range(IMG_CHANNELS)])
])
mnist_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
print(mnist_data[0][0].shape,mnist_data[0][0].max())
# network
G_net = Generator(Z_DIM, IMG_CHANNELS, IMG_SIZE, FEATURES_D).apply(weights_init).to(device)
D_net = Discriminator(IMG_CHANNELS, IMG_SIZE, FEATURES_D).apply(weights_init).to(device)


# Binary Cross Entropy loss
loss_fn = nn.BCELoss().to(device)

# Adam optimizer
optimizerG = optim.Adam(G_net.parameters(), lr=LR, betas=(0.5,0.999))
optimizerD = optim.Adam(D_net.parameters(), lr=LR, betas=(0.5,0.999))

losses = {}
losses['D_losses'] = []
losses['G_losses'] = []
for  epoch in range(TRAIN_EPOCHS):
    D_losses = []
    G_losses = []
    for step,(real_img, _) in enumerate(train_loader):
        # train discriminator D
        mini_batch = real_img.size()[0]
        label_r = torch.full((mini_batch, 1), real_label, dtype=torch.float32, device=device)
        label_f = torch.full((mini_batch, 1), fake_label, dtype=torch.float32, device=device)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        real_img = real_img.to(device)
        D_result = D_net(real_img).reshape(-1,1)
        errD_real = loss_fn(D_result, label_r)

        noise = torch.randn((mini_batch, Z_DIM, 1, 1)).to(device)
        fake_img = G_net(noise).detach()
        D_result = D_net(fake_img).reshape(-1,1)
        errD_fake = loss_fn(D_result, label_f)
        D_train_loss = errD_real + errD_fake

        optimizerD.zero_grad()
        D_train_loss.backward()
        optimizerD.step()
        D_losses.append(D_train_loss.item())

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        noise = torch.randn((mini_batch, Z_DIM, 1, 1)).to(device)
        fake_img = G_net(noise)
        D_result = D_net(fake_img).reshape(-1,1)
        G_train_loss = loss_fn(D_result, label_r)
        optimizerG.zero_grad()
        G_train_loss.backward()
        optimizerG.step()
        G_losses.append(G_train_loss.item())

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), TRAIN_EPOCHS, torch.FloatTensor(D_losses).mean(), torch.FloatTensor(G_losses).mean()))
    losses['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    losses['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    save_image(fake_img.data[:25], rand_path + '/epoch_{:04d}.png'.format(epoch), nrow=5, normalize=True)
    fix_result = G_net(fixed_z)
    save_image(fix_result.data[:25], fix_path + '/epoch_{:04d}.png'.format(epoch), nrow=5, normalize=True)
print("Training finish!... save training results")
torch.save(G_net.state_dict(), f"result/{project}/generator_param.pkl")
torch.save(D_net.state_dict(), f"result/{project}/discriminator_param.pkl")

#保存训练loss
with open(f'result/{project}/train_hist.pkl', 'wb') as f:
    pickle.dump(losses, f)
#训练loss
show_train_loss(losses, save=True, path=f'result/{project}/MNIST_GAN_train_loss.png')

# eval
G_net.load_state_dict(torch.load(f"result/{project}/generator_param.pkl"))
result = G_net(torch.randn((25, Z_DIM, 1, 1)).to(device))
save_image(result.data[:25], f'result/{project}/eval.png', nrow=5, normalize=True)
# show_generate()

