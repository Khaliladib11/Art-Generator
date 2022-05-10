import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

from torch.utils.data import Dataset

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


class Interface(Dataset):

    def __init__(self, data_path, size=(64, 64)):
        self.data_path = data_path
        self.size = size
        self.images = self.__load_images()

    def __load_images(self):
        images = []
        for image_folder in os.listdir(self.data_path):
            folder = os.path.join(self.data_path, image_folder)
            for image in os.listdir(folder):
                root, extension = os.path.splitext(image)
                if extension == '.jpg':
                    images.append(image)
                    # print(image)
        return images

    def apply_transforms(self, image):
        image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return image_transform(image)

    def get_image(self, idx, transform=True):
        image_file = self.images[idx]
        image_folder = image_file.split('_')[:-1]
        image_folder = '_'.join(image_folder)
        image = Image.open(os.path.join(self.data_path, image_folder, image_file))
        if transform:
            image = self.apply_transforms(image)

        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.get_image(item)
        return image


class GAN:

    def __init__(self, batch_size=128, latent_size=150, number_images=10, generated_image_path='../data/Generated_GAN'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.number_images = number_images
        self.generated_image_path = generated_image_path
        self.discriminator = self.get_discriminator().to(self.device)
        self.generator = self.get_generator().to(self.device)
        self.fixed_latent = torch.randn(64, latent_size, 1, 1, device=self.device)
        self.create_generated_images_folders()
        self.stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    def create_generated_images_folders(self):
        for i in range(self.number_images):
            os.makedirs(os.path.join(self.generated_image_path, f'image_{i}'), exist_ok=True)

    def get_discriminator(self):
        discriminator = nn.Sequential(
            # in: 3 x 64 x 64

            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid())
        return discriminator

    def get_generator(self):
        generator = nn.Sequential(
            # in: latent_size x 1 x 1

            nn.ConvTranspose2d(self.latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 64 x 64
        )
        return generator

    def train_discriminator(self, real_images, opt_d):
        # Clear discriminator gradients
        opt_d.zero_grad()

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=self.device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score

    def train_generator(self, opt_g):
        # Clear generator gradients
        opt_g.zero_grad()

        # Generate fake images
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        # Try to fool the discriminator
        preds = self.discriminator(fake_images)
        targets = torch.ones(self.batch_size, 1, device=self.device)
        loss = F.binary_cross_entropy(preds, targets)

        # Update generator weights
        loss.backward()
        opt_g.step()

        return loss.item()

    def denorm(self, img_tensors):
        return img_tensors * self.stats[1][0] + self.stats[0][0]

    def save_samples(self, epoch, latent_tensors):
        fake_images = self.generator(latent_tensors)
        for i in range(self.number_images):
            save_path = os.path.join(self.generated_image_path, f'image_{i}', f'image_{i}_{epoch}.jpg')
            plt.imsave(save_path, self.denorm(fake_images[i].permute(1, 2, 0).cpu().detach().numpy()))

    def fit(self, loader, epochs, lr):
        torch.cuda.empty_cache()

        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        # Create optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(epochs):
            for real_images in tqdm(loader):
                real_images = real_images.to(self.device)
                # Train discriminator
                loss_d, real_score, fake_score = self.train_discriminator(real_images, opt_d)
                # Train generator
                loss_g = self.train_generator(opt_g)

            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

            # Save generated images
            self.save_samples(epoch, self.fixed_latent)

        return losses_g, losses_d, real_scores, fake_scores
