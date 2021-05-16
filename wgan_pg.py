import os,sys
import matplotlib.pyplot as plt
import itertools
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
from torchvision.utils import save_image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# training parameters
project = 'wgan_gp'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
IMAGE_SIZE = 64
FEATURES_GEN = 16 ######## 16更好训练
FEATURES_CRITIC = 16
CHANNELS_IMG =1
Z_DIM = 100
lr = 1e-4
train_epoch = 20
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
fixed_z_ = torch.randn((5 * 5, Z_DIM)).view(-1, Z_DIM, 1, 1).to(device)    # fixed noise

if_show = True
if_save = True

transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
        ])
mnist_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_data,
                                           batch_size=batch_size,
                                           shuffle=True)
print(mnist_data[0][0].shape,'mnist_data max',mnist_data[0][0].max(),mnist_data[0][0].min())

# results save folder
if not os.path.isdir(f'result/{project}'):
    os.mkdir(f'result/{project}')
if not os.path.isdir(f'result/{project}/Random_results'):
    os.mkdir(f'result/{project}/Random_results')
if not os.path.isdir(f'result/{project}/Fixed_results'):
    os.mkdir(f'result/{project}/Fixed_results')



class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, features_g, channels_noise, channels_img):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            nn.init.constant_(m.bias, 0)


def show_result(message,noise,if_save=if_save):
    generated_image = netG(noise).detach().cpu().numpy()
    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].imshow(generated_image[k, 0,:], cmap='gray')
    plt.suptitle(message, fontsize=20)
    if if_show:
        plt.pause(0.01)
    if if_save:
        try:
            plt.savefig('result/{}/{:04d}_{:04d}.png'.format(project, epoch, batch_id), bbox_inches='tight')
        except IOError:
            print(IOError)

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

def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    BATCH_SIZE, C, H, W = real_samples.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1), device=device).repeat(1, C, H, W)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    mix_scores = critic(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad( #除去梯度，即为标量对矩阵求梯度
        outputs=mix_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(mix_scores,device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # print(gradients.shape)# [64, 1, 32, 32]
    gradients = gradients.view(gradients.shape[0],-1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() # norm 返回指定维度上的P范数
    return gradient_penalty
def gradient_penalty(critic, real, fake):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1),device=device).repeat(1, C, H, W)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
# plt.ion()
# network
netG = Generator(FEATURES_GEN, Z_DIM, CHANNELS_IMG).apply(weights_init).to(device)
netC = Critic(CHANNELS_IMG, FEATURES_CRITIC).apply(weights_init).to(device)
#  the progression of the generator
fixed_noise = torch.randn([32, Z_DIM, 1, 1], dtype=torch.float32, device=device)
# Establish convention for real and fake labels during training
# real_label = 1.
# fake_label = 0.

# Binary Cross Entropy loss
# loss_fn = nn.BCELoss().to(device)
# optimizer
# optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
# optimizerC = optim.RMSprop(netC.parameters(), lr=lr)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.0,0.9)) # beta的设置非常关键
optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(0.0,0.9))
losses = {}
losses['D_losses'] = []
losses['G_losses'] = []

if if_show:
    plt.ion()
if if_show or if_save:
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)


# train
for epoch in range(train_epoch):
    for batch_id, (data, target) in enumerate(train_loader):
        ############################
        # (1) Update C network: maximize Epr[C(x)] - C(G(z))
        ###########################
        for p in netC.parameters():  # reset requires_grad
            p.requires_grad = True # 梯度不通过权重传递，可以更改
        real_img = data.to(device)
        bs_size = real_img.shape[0]
        # r_label = torch.full((bs_size, 1, 1, 1), real_label, dtype=torch.float32, device=device)
        # f_label = torch.full((bs_size, 1, 1, 1), fake_label, dtype=torch.float32, device=device)
        for _ in range(CRITIC_ITERATIONS):

            noise = torch.randn([bs_size, Z_DIM, 1, 1], dtype=torch.float32, device=device)
            fake_img = netG(noise)
            fake_out = netC(fake_img.detach())#.reshape(-1)## #
            real_out = netC(real_img)#.reshape(-1)
            gp = compute_gradient_penalty(netC, real_img,fake_img)
            # gp = gradient_penalty(netC, real_img,fake_img)
            loss_C = (-(real_out.mean() - fake_out.mean()) + LAMBDA_GP * gp)

            optimizerC.zero_grad()
            loss_C.backward(retain_graph = True)
            optimizerC.step()
            # for param in netC.parameters():
            #     param.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        losses['D_losses'].append(loss_C.item())

        ############################
        # (2) Update G network: minimize Epr[C(x)] - C(G(z))
        ###########################
        for p in netC.parameters():
            p.requires_grad = False
        noise = torch.randn([bs_size, 100, 1, 1], dtype=torch.float32, device=device)
        fake_img = netG(noise)
        output = netC(fake_img)#.reshape(-1)
        loss_G = -output.mean()

        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()

        losses['G_losses'].append(loss_G.item())

        ############################
        # visualize
        ###########################

        if batch_id % 100 == 0:
            msg = 'Epoch ID={0} Batch ID={1} \n C-Loss={2} G-Loss={3}'.format(epoch, batch_id, loss_C.item(),
                                                                              loss_G.item())
            print(msg)
            # netC.eval()
            # netG.eval()
            show_result(msg, fixed_noise)  # noise
            # netC.train()
            # netG.train()

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




