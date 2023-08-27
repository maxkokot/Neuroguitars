import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import os


def weights_init(m):
    '''the function for weights initialization
    has been taken from there
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_model(trainer, model_dir, model_name='model.ckpt'):
    path = os.path.join(model_dir, model_name)
    trainer.save_checkpoint(path)


def load_model(model_dir, img_size, latent_dim, lr,
               model_name='model.ckpt'):
    path = os.path.join(model_dir, model_name)
    model = DCGAN.load_from_checkpoint(checkpoint_path=path,
                                       img_size=img_size,
                                       latent_dim=latent_dim,
                                       lr=lr)
    return model


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.start_size = self.img_size // 16
        self.fc_layer = nn.Sequential(
            nn.Linear(latent_dim, 512 * (self.start_size ** 2)),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 4, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 4, stride=1, padding=2),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 4, stride=1, padding=2),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.fc_layer(x)
        out = out.view(out.shape[0], 512,
                       self.start_size,
                       self.start_size)
        out = self.conv_layers(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.fin_size = self.img_size // 16
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(512 * (self.fin_size ** 2), 1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.shape[0], -1)
        out = self.fc_layer(out)
        return out


class DCGAN(pl.LightningModule):

    def __init__(self,
                 img_size,
                 latent_dim=128,
                 lr=0.0002,
                 beta_1=0.5,
                 beta_2=0.999,
                 **kwargs):
        super().__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.generator = Generator(latent_dim=self.latent_dim,
                                   img_size=self.img_size)
        self.discriminator = Discriminator(img_size=self.img_size)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        z = torch.randn(len(real_imgs), self.latent_dim).type_as(real_imgs)
        fake_imgs = self(z)

        if optimizer_idx == 0:

            y_true = torch.ones(len(z), 1).type_as(real_imgs)
            y_pred = self.discriminator(fake_imgs)
            g_loss = self.adversarial_loss(y_pred, y_true)
            self.log_dict({
                        'g_loss': g_loss,
                    })
            return g_loss

        elif optimizer_idx == 1:

            y_true_r = torch.ones(len(real_imgs), 1).type_as(real_imgs)
            y_pred_r = self.discriminator(real_imgs)
            real_loss = self.adversarial_loss(y_pred_r, y_true_r)

            y_true_f = torch.zeros(len(real_imgs), 1).type_as(real_imgs)
            y_pred_f = self.discriminator(fake_imgs.detach())
            fake_loss = self.adversarial_loss(y_pred_f, y_true_f)

            d_loss = (real_loss + fake_loss) / 2

            with torch.no_grad():
                real_acc = torch.round(y_pred_r).sum() / len(real_imgs)
                fake_acc = 1 - torch.round(y_pred_f).sum() / len(fake_imgs)

            d_loss = (real_loss + fake_loss) / 2
            self.log_dict({
                        'd_loss': d_loss,
                        'fake_loss': fake_loss,
                        'real_loss': real_loss,
                        'fake_acc': fake_acc,
                        'real_acc': real_acc,
                    })
            return d_loss

    def configure_optimizers(self):

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr,
                                 betas=(self.beta_1, self.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr,
                                 betas=(self.beta_1, self.beta_2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):

        z = torch.randn(8, self.latent_dim).to(self.device)
        imgs = self(z)
        grid = torchvision.utils.make_grid(imgs)
        self.logger.experiment.add_image('guitars', grid, self.current_epoch)
