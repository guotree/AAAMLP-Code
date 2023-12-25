import os
import torch
import numpy as np
from PIL import Image, ImageFile
from collections import defaultdict
from torchvision import transforms
from albumentations import (
    Compose,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
TRAIN_PATH = "/kaggle/input/siim-png-images/train_png/"

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_ids,
            transform=True,
            preprocessing_fn=None
    ):
        self.data = defaultdict(dict)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0,
                    rotate_limit=0,
                    p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 100)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.1
                        )
                    ],
                    p=0.5
                )
            ]
        )
        for imgid in image_ids:
            self.data[imgid] = {
                "img_path": os.path.join(TRAIN_PATH, imgid + ".png"),
                "mask_path": os.path.join(TRAIN_PATH, imgid + "_mask.png")
            }

    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        img_path = self.data[item]["img_path"]
        mask_path = self.data[item]["mask_path"]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        mask = Image.open(mask_path).convert("RGB")
        mask = np.array(mask)
        mask = (mask>=1).astype("float32")
        if self.transform is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        img = self.preprocessing_fn(img)
        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float()
        }