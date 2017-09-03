import os
import os.path
import torch
import pandas as pd

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = ['.png', '.jpg']


def default_inception_transform(img_size):
    tf = transforms.Compose([
        transforms.Scale(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        LeNormalize(),
    ])
    return tf



class Dataset(data.Dataset):

    def __init__(self,imglist,transform=None):

        if len(imglist) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imglist
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        target = None
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.imgs)

