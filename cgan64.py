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
from torchvision.utils import save_image, make_grid
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# training parameters
project = 'cgan'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMAGE_SIZE = 64
FEATURES_GEN = 32#16
FEATURES_CRITIC = 32#16
CHANNELS_IMG =1
NUM_CLASSES = 10
GEN_EMBEDDING_SIZE = 100
IMG_CHANNELS = 1
Z_DIM = 100
lr = 1e-4
TRAIN_EPOCHS = 20
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
fixed_noise = torch.randn([10, Z_DIM, 1, 1], dtype=torch.float32, device=device)
fixed_labels = torch.arange(10, device=device)

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
mnist_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
print(mnist_data[0][0].shape,'mnist_data max',mnist_data[0][0].max(),mnist_data[0][0].min())



class Critic(nn.Module):
    def __init__(self, features_d, channels_img, num_classes, img_size):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # input: N x (channels_img+1) x 64 x 64 -> 32x32
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),# 32x32 -> 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),# 16x16 -> 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),# 8x8 -> 4x4
            # 4x4 -> 1x1
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes,img_size*img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False,),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x,labels):
        embedding = self.embed(labels).view(labels.shape[0],1,self.img_size,self.img_size)
        x = torch.cat([x,embedding],dim=1)
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, features_g, channels_noise, channels_img, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False,),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(),
        )

    def forward(self, x,labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x,embedding],dim=1)
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

def compute_gradient_penalty(critic, labels, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    BATCH_SIZE, C, H, W = real_samples.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1), device=device).repeat(1, C, H, W)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    # critic scores
    mix_scores = critic(interpolates, labels)
    gradients = autograd.grad(
        outputs=mix_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(mix_scores,device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.shape[0],-1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() # norm 返回指定维度上的P范数
    return gradient_penalty

netG = Generator(FEATURES_GEN, Z_DIM, CHANNELS_IMG, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING_SIZE).apply(weights_init).to(device)
netC = Critic(FEATURES_CRITIC, CHANNELS_IMG, NUM_CLASSES, IMAGE_SIZE).apply(weights_init).to(device)
#  the progression of the generator



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



# train
for epoch in range(TRAIN_EPOCHS):
    D_losses = []
    G_losses = []
    for step, (data, target) in enumerate(train_loader):
        #  Update C network: maximize Epr[C(x)] - C(G(z))
        real_img = data.to(device)
        target = target.to(device)
        bs_size = real_img.shape[0]

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn([bs_size, Z_DIM, 1, 1], dtype=torch.float32, device=device)
            fake_img = netG(noise,target)#.detach()
            fake_out = netC(fake_img,target).reshape(-1)
            real_out = netC(real_img,target).reshape(-1)
            gp = compute_gradient_penalty(netC,target, real_img,fake_img)
            loss_C = (-(real_out.mean() - fake_out.mean()) + LAMBDA_GP * gp)
            optimizerC.zero_grad()
            loss_C.backward(retain_graph = True)
            optimizerC.step()
            D_losses.append(loss_C.item())



        # Update G network: minimize Epr[C(x)] - C(G(z))
        # noise = torch.randn([bs_size, 100, 1, 1], dtype=torch.float32, device=device)
        # fake_img = netG(noise)
        output = netC(fake_img,target).reshape(-1)
        loss_G = -output.mean()
        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()
        G_losses.append(loss_G.item())

    losses['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    losses['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), TRAIN_EPOCHS, torch.FloatTensor(losses['D_losses']).mean(),
        torch.FloatTensor(losses['G_losses']).mean()))
    save_image(fake_img.data[:25], rand_path + '/epoch_{:04d}.png'.format(epoch), nrow=5, normalize=True)
    fix_result = netG(fixed_noise,fixed_labels)
    save_image(fix_result.data[:25], fix_path + '/epoch_{:04d}.png'.format(epoch), nrow=5, normalize=True)
# 保存训练loss
with open(f"result/{project}/train_hist.pkl", 'wb') as f:
    pickle.dump(losses, f)
# 画loss
show_train_loss(losses, save=True, path=f"result/{project}/MNIST_GAN_train_hist.png")

print("Training finish!... save training results")
if not os.path.isdir(f'result/{project}/checkpoints'):
    os.mkdir(f'result/{project}/checkpoints')
torch.save(netG.state_dict(), f"result/{project}/generator_param.pkl")
torch.save(netC.state_dict(), f"result/{project}/critic_param.pkl")


# eval
netG.load_state_dict(torch.load(f"result/{project}/generator_param.pkl"))
result = netG(torch.randn((10, Z_DIM, 1, 1)).to(device),fixed_labels)
save_image(result.data[:10], f'result/{project}/eval.png', nrow=5, normalize=True)

