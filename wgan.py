import os,sys
import matplotlib.pyplot as plt
import itertools
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# training parameters
project = 'wgan'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 0.00005
train_epoch = 10
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01
Z_DIM = 100
fixed_z_ = torch.randn((5 * 5, Z_DIM)).view(-1, Z_DIM, 1, 1).to(device)    # fixed noise
# data_loader
IMAGE_SIZE = 32#64#32#
IMG_CHANNELS = 1

# results save folder
rand_path = f'result/{project}/Random_results'
fix_path = f'result/{project}/Fixed_results'
if not os.path.isdir(f'result/'):
    os.mkdir(f'result/')
if not os.path.isdir(f'result/{project}'):
    os.mkdir(f'result/{project}')
if not os.path.isdir(rand_path):
    os.mkdir(rand_path)
if not os.path.isdir(fix_path):
    os.mkdir(fix_path)

transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5 for _ in range(IMG_CHANNELS)],
                         std=[0.5 for _ in range(IMG_CHANNELS)])
        ])
mnist_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
print(mnist_data[0][0].shape,'mnist_data max',mnist_data[0][0].max(),mnist_data[0][0].min())



class Generator(nn.Module):
    # initializers
    def __init__(self, features_d=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input is Z, [B, 100, 1, 1] -> [B, 64 * 4, 4, 4]
            nn.ConvTranspose2d(100, features_d * 4, 4, 1, 0, bias=False),
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
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.gen(input)

class Critic(nn.Module):
    # initializers
    def __init__(self, features_d=64):
        super(Critic, self).__init__()
        self.dis = nn.Sequential(
            # input [B, 1, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(1, features_d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d),
            nn.LeakyReLU(0.1),
            # state size. [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.1),
            # state size. [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(features_d*2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.1),

            # state size. [B, 256, 4, 4] -> [B, 1, 1, 1]
            nn.Conv2d(features_d * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

        )

    # forward method
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


# plt.ion()
# network
netG = Generator(64).apply(weights_init).to(device)
netC = Critic(64).apply(weights_init).to(device)
#  the progression of the generator
fixed_noise = torch.randn([32, 100, 1, 1], dtype=torch.float32, device=device)
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Binary Cross Entropy loss
# loss_fn = nn.BCELoss().to(device)
# optimizer
optimizerG = optim.RMSprop(netG.parameters(), lr=LR)
optimizerC = optim.RMSprop(netC.parameters(), lr=LR)
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5,0.999)) # beta的设置非常关键
# optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(0.5,0.999))
losses = {}
losses['D_losses'] = []
losses['G_losses'] = []



# train
for epoch in range(train_epoch):
    for batch_id, (data, target) in enumerate(train_loader):
        # Update C network: maximize Epr[C(x)] - C(G(z))
        for p in netC.parameters():  # reset requires_grad
            p.requires_grad = True
        real_img = data.to(device)
        bs_size = real_img.shape[0]
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn([bs_size, Z_DIM, 1, 1], dtype=torch.float32, device=device)
            fake_img = netG(noise)
            fake_out = netC(fake_img.detach() ).reshape(-1) # 不需要detach
            real_out = netC(real_img).reshape(-1)
            loss_C = -(real_out.mean() - fake_out.mean())
            optimizerC.zero_grad()
            loss_C.backward() # retain_graph = True
            optimizerC.step()
            for param in netC.parameters():
                param.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        losses['D_losses'].append(loss_C.item())

        ############################
        # (2) Update G network: minimize Epr[C(x)] - C(G(z))
        ###########################
        for p in netC.parameters():
            p.requires_grad = False
        noise = torch.randn([bs_size, 100, 1, 1], dtype=torch.float32, device=device)
        fake_img = netG(noise)
        output = netC(fake_img)
        loss_G = -output.mean()
        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()

        losses['G_losses'].append(loss_G.item())

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), train_epoch, torch.FloatTensor(losses['D_losses']).mean(),
        torch.FloatTensor(losses['G_losses']).mean()))

# 保存训练loss
with open(f"result/{project}/train_hist.pkl", 'wb') as f:
    pickle.dump(losses, f)
# 画loss
show_train_loss(losses, save=True, path=f"result/{project}/MNIST_GAN_train_hist.png")

print("Training finish!... save training results")
if not os.path.isdir(f'result/{project}/checkpoints'):
    os.mkdir(f'result/{project}/checkpoints')
torch.save(netG.state_dict(), f"result/{project}/checkpoints/generator_param.pkl")
torch.save(netC.state_dict(), f"result/{project}/checkpoints/critic_param.pkl")

# eval
netG.load_state_dict(torch.load(f"result/{project}/generator_param.pkl"))
result = netG(torch.randn((25, Z_DIM, 1, 1)).to(device))
save_image(result.data[:25], fix_path + '/eval.png', nrow=5, normalize=True)