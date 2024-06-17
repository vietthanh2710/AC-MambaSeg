import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image

class RandomCrop(transforms.RandomResizedCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        for imgCount in range(len(imgs)):
            imgs[imgCount] = transforms.functional.resized_crop(imgs[imgCount], i, j, h, w, self.size, self.interpolation)
        return imgs
class ISICLoader(Dataset):
    def __init__(self, images, masks,
                 transform=True, typeData = "train"):
        self.transform = transform if typeData == "train" else False  # augment data bool
        self.typeData = typeData
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)

    def rotate(self, image, mask, degrees=(-15,15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask
    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask
    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask
    def random_resized_crop(self, image, mask, p=0.1):
        if torch.rand(1) < p:
            image, mask = RandomCrop((192, 256), scale=(0.8, 0.95))([image, mask])
        return image, mask

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        return image, mask

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx])
    ####################### augmentation data ##############################
        if self.transform:
            image, mask = self.augment(image, mask)
        image = transforms.ToTensor()(image)
        mask = np.asarray(mask, np.int64)
        mask = torch.from_numpy(mask[np.newaxis])
        return image, mask
    
