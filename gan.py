import os
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image



project = 'gan'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Z_DIM = 64
BATCH_SIZE = 128
LR = 0.0002
TRAIN_EPOCHS = 100
fixed_z = torch.randn((5 * 5, Z_DIM)).to(device)    # fixed noise
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

# G(z)
class Generator(nn.Module):
    # initializers
    def __init__(self, z_dim=32, img_dim=28*28):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    # forward method
    def forward(self, input):
        return self.gen(input)

class Discriminator(nn.Module):
    # initializers
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    # forward method
    def forward(self, input):
        return self.disc(input)


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
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
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
])
mnist_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
print(mnist_data[0][0].shape,mnist_data[0][0].max())
# network
G_net = Generator(z_dim=Z_DIM, img_dim=28*28).apply(weights_init).to(device)
D_net = Discriminator(input_size=28*28).apply(weights_init).to(device)


# Binary Cross Entropy loss
loss_fn = nn.BCELoss().to(device)

# Adam optimizer
G_optimizer = optim.Adam(G_net.parameters(), lr=LR)
D_optimizer = optim.Adam(D_net.parameters(), lr=LR)



losses = {}
losses['D_losses'] = []
losses['G_losses'] = []
for  epoch in range(TRAIN_EPOCHS):
    D_losses = []
    G_losses = []
    for step,(x, _) in enumerate(train_loader):
        # train discriminator D
        mini_batch = x.size()[0]
        r_label = torch.full((mini_batch, 1), real_label, dtype=torch.float32, device=device)
        f_label = torch.full((mini_batch, 1), fake_label, dtype=torch.float32, device=device)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # D on real
        x = x.view(-1, 28 * 28).to(device)
        D_result = D_net(x)
        D_real_loss = loss_fn(D_result, r_label)
        # D_real_score = D_result

        # D on fake
        z = torch.randn((mini_batch, Z_DIM)).to(device)
        x_fake = G_net(z)
        D_result = D_net(x_fake)
        D_fake_loss = loss_fn(D_result, f_label)
        # D_fake_score = D_result
        D_train_loss = D_real_loss + D_fake_loss
        D_net.zero_grad()
        D_train_loss.backward()
        D_optimizer.step()
        D_losses.append(D_train_loss.item())
        # print(type(D_train_loss.item()))# <class 'float'>

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        z = torch.randn((mini_batch, Z_DIM)).to(device)
        x_fake = G_net(z)
        D_result = D_net(x_fake)
        G_train_loss = loss_fn(D_result, r_label)
        G_net.zero_grad()
        G_train_loss.backward()
        G_optimizer.step()
        G_losses.append(G_train_loss.item())

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), TRAIN_EPOCHS, torch.FloatTensor(D_losses).mean(), torch.FloatTensor(G_losses).mean()))
    losses['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    losses['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    rand_result = x_fake.reshape(-1,1,28,28)
    # save_image(G_result.data[:25], f"result/{project}/%d.png" % epoch, nrow=5, normalize=True)
    save_image(rand_result.data[:25], rand_path + '/epoch_{:04d}.png'.format(epoch), nrow=5, normalize=True)
    fix_result = G_net(fixed_z).reshape(-1, 1, 28, 28)
    save_image(fix_result.data[:25], fix_path + '/epoch_{:04d}.png'.format(epoch), nrow=5, normalize=True)
print("Training finish!... save training results")
torch.save(G_net.state_dict(), f"result/{project}/generator_param.pkl")
torch.save(D_net.state_dict(), f"result/{project}/discriminator_param.pkl")

#保存训练loss
with open(f'result/{project}/train_loss.pkl', 'wb') as f:
    pickle.dump(losses, f)
#训练loss
show_train_loss(losses, save=True, path=f'result/{project}/MNIST_GAN_train_loss.png')

# eval
G_net.load_state_dict(torch.load(f"result/{project}/generator_param.pkl"))
result = G_net(torch.randn((25, Z_DIM)).to(device)).reshape(-1, 1, 28, 28)
save_image(result.data[:25], f'result/{project}/eval.png', nrow=5, normalize=True)
# show_generate()

