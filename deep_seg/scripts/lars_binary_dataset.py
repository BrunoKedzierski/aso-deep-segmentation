import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import InterpolationMode
from torchvision import tv_tensors


class WaterBinaryDataset(Dataset):
    def __init__(self, root, split, image_size, augmentations=None):
        self.split = split
        self.image_size = image_size
        self.augmentations = augmentations

        self.image_dir = Path(root) / split / "images"
        self.mask_dir = Path(root) / split / "semantic_masks"

        self.image_paths = sorted(self.image_dir.glob("*.jpg"))

        self.has_masks = self.mask_dir.exists() and len(list(self.mask_dir.glob("*.png"))) > 0

        if self.has_masks:
            self.mask_paths = {p.stem: p for p in self.mask_dir.glob("*.png")}
            self.pairs = [
                (img, self.mask_paths[img.stem])
                for img in self.image_paths
                if img.stem in self.mask_paths
            ]
        else:
            # Test images
            self.pairs = self.image_paths

        self.resize = T.Resize(
            size=(image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        print(f"{split}: {len(self.pairs)} samples | masks: {self.has_masks}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        # Test Images
        if not self.has_masks:
            img_path = self.pairs[idx]
            image = read_image(str(img_path)).float() / 255.0
            image = tv_tensors.Image(image)

            image = self.resize(image)

            if self.augmentations is not None:
                image = self.augmentations(image)

            return image.as_subclass(torch.Tensor)

        # Train and val images
        img_path, mask_path = self.pairs[idx]

        image = read_image(str(img_path)).float() / 255.0
        mask = read_image(str(mask_path))[0].long()

        mask[mask == 255] = 0
        mask = (mask == 1).float()
        mask = mask.unsqueeze(0)

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        image, mask = self.resize(image, mask)

        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)

        return image.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor)