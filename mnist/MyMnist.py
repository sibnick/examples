from typing import Tuple, Any, Optional, Callable
import numpy as np
import torch
from PIL import Image

from torchvision import datasets

class MyMnist(datasets.MNIST):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            checker=None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform, train=train, download=download)
        self.checker = checker

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(index, img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
