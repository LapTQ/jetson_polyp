import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.nn.functional import one_hot
import albumentations as A
import numpy as np
import os
import random
from pathlib import Path


HERE = Path(__file__).parent


class PolypDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, in_size, out_size, transform=None, n=4096):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.in_size = in_size
        self.out_size = out_size
        self.n = n
        self.transform = transform

        self.img_names = os.listdir(img_dir)

        self.lbl_ext = os.path.splitext(os.listdir(lbl_dir)[0])[1]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sample = random.choices(self.img_names, k=self.n)
        name, ext = os.path.splitext(sample[idx])
        img_path = os.path.join(self.img_dir, name + ext)
        lbl_path = os.path.join(self.lbl_dir, name + self.lbl_ext)

        # RGB
        img = cv2.imread(img_path)[:, :, ::-1]
        lbl = cv2.imread(lbl_path, 0)

        if self.transform:
            transformed = self.transform(image=img, mask=lbl)
            img = transformed['image']
            lbl = transformed['mask']

        img = A.Resize(self.in_size, self.in_size)(image=img)['image']
        lbl = A.Resize(self.out_size, self.out_size)(image=img, mask=lbl)['mask']

        lbl = np.expand_dims(lbl, axis=0).astype('float32') / 255.

        # uint8 [0, 255] (h, w, c) RGB to float [0., 1.] (c, h, w) RGB
        img = ToTensor()(img.copy())
        lbl = torch.from_numpy(lbl)

        return img, lbl


def get_dataloader(**kwargs):
    dataset = PolypDataset(
        img_dir=kwargs['img_dir'],
        lbl_dir=kwargs['lbl_dir'],
        in_size=kwargs['in_size'],
        out_size=kwargs['out_size'],
        transform=kwargs['transform']
    )
    dataloader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=kwargs['shuffle'])
    return dataloader


if __name__ == '__main__':

    dataloader = get_dataloader(img_dir=str(HERE/'../../../data/Neo-final/images'), lbl_dir=str(HERE/'../../../data/Neo-final/mask_images'), batch_size=4, in_size=224, out_size=210, transform=None, shuffle=True)
    images, labels = next(iter(dataloader))
    import matplotlib.pyplot as plt
    plt.imshow(labels[0][0].float())
    plt.show()
