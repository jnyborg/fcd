from glob import glob

from albumentations import HorizontalFlip, Normalize, Compose
from tifffile import tifffile
from torch.utils import data
from PIL import Image
import torch
import os
import random
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2


class PatchDataset(data.Dataset):
    def __init__(self, x, patch_size, crop_size, transforms):
        assert x.dtype == np.uint16
        self.x = x
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.transforms = transforms
        raster_height, raster_width, _ = x.shape

        # Create a raster with height and width divisible by 'cropped_tile_size', and pad the raster with CROP_SIZE.
        cropped_patch_size = patch_size - crop_size * 2
        pad_height = (self.crop_size, (cropped_patch_size - raster_height % cropped_patch_size) + self.crop_size)
        pad_width = (self.crop_size, (cropped_patch_size - raster_width % cropped_patch_size) + self.crop_size)

        self.cropped_patch_size = cropped_patch_size

        self.x = np.pad(x, (pad_height, pad_width, (0, 0)), 'reflect')

        self.patches = []
        for row in range(0, raster_height, cropped_patch_size):
            for col in range(0, raster_width, cropped_patch_size):
                self.patches.append((row, col))

    def __getitem__(self, index):
        row, col = self.patches[index]
        image = self.x[row:row + self.patch_size, col:col + self.patch_size]
        return {'image': self.transforms(image=image)['image'], 'row': row, 'col': col}

    def __len__(self):
        return len(self.patches)


class L8BiomeDataset(data.Dataset):
    def __init__(self, root, transform, mode='train', mask_file='mask.tif', keep_ratio=1.0, only_cloudy=False):
        self.root = root = os.path.join(root, mode)
        classes, class_to_idx = self._find_classes(root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.images = self._make_dataset(root, class_to_idx)

        if only_cloudy:
            self.images = [img for img in self.images if img[1] == 1]

        if keep_ratio < 1.0:
            # Subsample images for supervised training on fake images, and fine-tuning on keep_ratio% real images
            print('Dataset size before keep_ratio', len(self.images))
            random.seed(42)  # Ensure we pick the same 1% across experiments
            random.shuffle(self.images)
            self.images = self.images[:int(keep_ratio * len(self.images))]
            print('Dataset size after keep_ratio', len(self.images))

        self.transform = transform
        self.return_mask = mask_file is not None
        self.mask_file = mask_file

    def __getitem__(self, index):
        patch_dir, label, patch_name = self.images[index]
        image = tifffile.imread(os.path.join(patch_dir, 'image.tif'))

        out = {
            'patch_name': patch_name,
            'label': torch.tensor(label).long(),
        }
        if self.return_mask:
            # 0 = invalid, 1 = clear, 2 = clouds
            mask = tifffile.imread(os.path.join(patch_dir, self.mask_file)).astype(np.long)
            sample = self.transform(image=image, mask=mask)
            out['image'] = sample['image']
            out['mask'] = sample['mask']
        else:
            out['image'] = self.transform(image=image)['image']
        return out

    def __len__(self):
        return len(self.images)

    def _make_dataset(self, root, class_to_idx):
        images = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(root, target)
            if not os.path.isdir(d):
                continue
            for patch_dir, _, file_names in sorted(os.walk(d)):
                if len(file_names) == 0:
                    continue

                patch_name = patch_dir.split('/')[-1]
                images.append((patch_dir, self.class_to_idx[target], patch_name))

        return images

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class L8SparcsDataset(data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.images = self._make_dataset(root)
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        image_path, mask_path = self.images[index]
        image = tifffile.imread(image_path)
        orig_mask = np.array(Image.open(mask_path))
        mask = np.zeros_like(orig_mask, dtype=np.uint8)
        # 0 Shadow, 1 Shadow over Water, 2 Water, 3 Snow, 4 Land, 5 Cloud, 6 Flooded
        mask[orig_mask == 5] = 1  # Only use 0 = background and 1 = cloud

        if self.mode == 'train':
            label = (mask == 1).any()
            return self.transform(image=image)['image'], torch.tensor([label]).float()
        else:
            return self.transform(image=image)['image'], mask


    def __len__(self):
        return len(self.images)

    def _make_dataset(self, root):
        dir = os.path.join(root, 'sending')
        datas = sorted(glob(os.path.join(dir, '*_data.tif')))
        masks = sorted(glob(os.path.join(dir, '*_mask.png')))
        return list(zip(datas, masks))

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def get_loader(image_dir, batch_size=16, dataset='L8Biome', mode='train',
               num_workers=4, num_channels=3, mask_file=None, keep_ratio=1.0, shuffle=None, force_no_aug=False, only_cloudy=False, pin_memory=True):
    """Build and return a data loader."""
    transform = []
    if mode == 'train' and not force_no_aug:
        transform.append(HorizontalFlip())
    transform.append(Normalize(mean=(0.5,) * num_channels, std=(0.5,) * num_channels, max_pixel_value=2 ** 16 - 1))
    transform.append(ToTensorV2())
    transform = Compose(transform)

    if dataset == 'L8Biome':
        dataset = L8BiomeDataset(image_dir, transform, mode, mask_file, keep_ratio, only_cloudy=only_cloudy)
    elif dataset == 'L8Sparcs':
        dataset = L8SparcsDataset(image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train') if shuffle is None else shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader
