# Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import glob
import cv2


# Natural Transfer Class

class NaturalTransfer:

    def __init__(self, data_path, generated_img_path, pretrained=True, weight_path=None, device=None):
        """
        Constructor for the Natural Transfer class
        :param data_path: path to where the data are available
        :param generated_img_path: path where we should put the data after generate art
        :param pretrained: Boolean params, to check if we want to load a pre-trained model
        :param weight_path: path to weights for the model, in case we don't want to load pre-trained model
        :param device: Device we want to use, 'cpu' or 'cuda'
        """
        self.data_path = data_path
        self.generated_img_path = generated_img_path

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            assert device in ['cpu', 'cuda'], "The available options for device are: 'cpu' and 'cuda'."
            self.device = device

        assert pretrained in [True, False], "pretrained is boolean parameter"
        assert (pretrained is True and weight_path is None), "You have to pass model's weight path."

        self.pretrained = pretrained
        self.weight_path = weight_path

        self.feature_extractor = self.__initialize_feature_extractor()
        self.feature_extractor.to(self.device)

    # method to initialize the feature extractor, in our case vgg19
    def __initialize_feature_extractor(self):
        if self.pretrained is True:
            feature_extractor = models.vgg19(pretrained=self.pretrained).features
        else:
            feature_extractor = models.vgg19(pretrained=False)
            feature_extractor.load_state_dict(torch.load(self.weight_path))
            feature_extractor = feature_extractor.features

        # we have to make sure that we won't update the vgg19 params in backpropagation
        for param in feature_extractor.parameters():
            param.requires_grad_(False)

        return feature_extractor

    # method to load an image
    def load_image(self, image_path, max_size=400, shape=None):
        # load image from path
        image = Image.open(os.path.join(self.data_path, image_path)).convert("RGB")

        # check the max size
        size = max_size if max(image.size) > max_size else max(image.size)

        if shape is not None:
            size = shape

        # apply some transforms to the image from torchvision
        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # add the batch size, which is 1
        image = img_transform(image)[:3, :, :].unsqueeze(0).to(self.device)
        return image.to(self.device)

    # static method to convert a tensor to normal Image object
    @staticmethod
    def convert_image(image_tensor):
        image = image_tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)
        return image

    # method to display the images
    def display_summary(self, tensors):
        fig, axes = plt.subplots(1, len(tensors), figsize=(len(tensors) * 10, 10))
        for i in range(len(tensors)):
            axes[i].imshow(self.convert_image(tensors[i]))
            axes[i].axis('off')

    # method yo extract the features we want from certain image using the model we are using
    def get_features(self, image):
        features = {}
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
        x = image
        for name, layer in self.feature_extractor._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    @staticmethod
    def gram_matrix(tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram

    # the fit method, to apply the transfer learning
    def fit(self,
            content_img,
            style_img,
            epochs,
            save_path,
            show_every=5,
            save_every=10,
            style_weights=None,
            alpha=1,
            beta=1e6,
            lr=0.003,
            ):
        if style_weights is None:
            style_weights = {'conv1_1': 1.,
                             'conv2_1': 0.8,
                             'conv3_1': 0.5,
                             'conv4_1': 0.3,
                             'conv5_1': 0.1}
        content_features = self.get_features(content_img)
        style_features = self.get_features(style_img)
        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}
        target = content_img.clone().requires_grad_(True).to(self.device)

        optimizer = optim.Adam([target], lr=lr)

        for epoch in range(epochs):
            target_features = self.get_features(target)
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

            style_loss = 0

            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (d * h * w)

            total_loss = alpha * content_loss + beta * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            img = self.convert_image(target)
            if epoch % show_every == 0:
                print(f"Epoch number: {epoch}, Total Loss: {total_loss.item()}")
                plt.imshow(img)
                plt.axis('off')
                plt.show()

            if epoch % save_every == 0:
                self.save_image(img, save_path, epoch)

        return img

    # method to save the image
    def save_image(self, image, save_path, epoch):
        if not os.path.exists(os.path.join(self.generated_img_path, save_path)):
            os.makedirs(os.path.join(self.generated_img_path, save_path))

        file_name = os.path.join(self.generated_img_path, save_path, save_path + f'_{epoch}.jpg')

        plt.imsave(file_name, image)

    # method to create a video from group of images
    def create_video(self, images_file_name, fps=300):
        video_path = os.path.join(self.generated_img_path, 'videos', f'{images_file_name}.avi')
        # print(video_path)
        images_path = os.path.join(self.generated_img_path, images_file_name)
        # print(len(os.listdir(images_path)))
        images = [img for img in os.listdir(images_path) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(images_path, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_path, 0, fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(images_path, image)))

        cv2.destroyAllWindows()
        video.release()
