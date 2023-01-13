import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return transforms.Compose([
        transforms.RandomCrop(crop_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply(transforms.RandomRotation(degrees=90, interpolation = transforms.InterpolationMode.BICUBIC), p=0.5),
        transforms.ToTensor()
    ])


def train_lr_transform(crop_size, upscale_factor):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size // upscale_factor, interpolation=Image.Resampling.BICUBIC),
        transforms.ToTensor()
    ])


class DS(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(DS, self).__init__()
        print('Setting up data...')

        self.image_filenames = []
        for ds_path in dataset_dir:
            for x in os.listdir(ds_path):
                self.image_filenames.append(os.path.join(ds_path, x))
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)