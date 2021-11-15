import argparse
import random
from collections import namedtuple, defaultdict
from itertools import product
from pathlib import Path
from typing import List

import cv2
import numpy as np
import rasterio
import tifffile
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


BANDS_30M = [
    'B1',   # coastal/aerosol
    'B2',  # blue
    'B3',  # green
    'B4',  # red
    'B5',   # nir
    'B6',   # swir1
    'B7',   # swir2
    'B9',   # cirrus
    'B10',  # tir1
    'B11'   # tir2
]
BANDS_15M = [
    # 'B8'  # panchromatic
]

Landsat8Image = namedtuple('Landsat8Image', 'name, biome, bands_15m, bands_30m, manual_cloud_mask qa_cloud_mask')


def prepare_patches(config):
    data_path = Path(config.data_path)
    output_path = Path(config.output_dir)
    patch_size = config.patch_size

    output_path.mkdir(exist_ok=True, parents=True)
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(exist_ok=True, parents=True)

    thumbnail_dir = output_path / 'thumbnails'
    thumbnail_dir.mkdir(exist_ok=True)

    images = get_landsat8_images(data_path)

    # Compute 3:1:2 train/val/test split, and save chosen assignment to file.
    train_val_test = split_train_val_test(images, val_ratio=2 / 12, test_ratio=4 / 12, seed=config.seed)

    with open(output_path / 'assignment.txt', mode='w') as f:
        for idx, split in enumerate(train_val_test):
            x = images[idx]
            line = "{},{},{}\n".format(split, x.biome, x.name)
            f.write(line)

    patch_ids = {'train': 0, 'val': 0, 'test': 0}

    for img_idx, image in enumerate(tqdm(images, desc='Reading L8 tile')):
        split = train_val_test[img_idx]
        split_dir = output_path / split
        x, mask = read_image(image)
        assert x.dtype == np.uint16

        height, width, _ = x.shape
        patches = list(product(range(0, patch_size * (height // patch_size), patch_size),
                               range(0, patch_size * (width // patch_size), patch_size)))

        # Create thumbnail of full image for debugging
        thumbnail = np.clip(1.5 * (x[..., [3, 2, 1]].copy() >> 8), 0, 255).astype(np.uint8)
        thumbnail = cv2.resize(thumbnail, (1000, 1000))
        Image.fromarray(thumbnail).save(
            str(thumbnail_dir / '{}_thumbnail_{}_{}.jpg'.format(split, image.biome, image.name)))

        if split is 'test':
            continue  # use raw images for testing instead of patches

        for row, col in patches:
            patch_x = x[row:row + patch_size, col:col + patch_size]
            patch_mask = mask[row:row + patch_size, col:col + patch_size]
            if (patch_mask == 0).all():  # ignore completely invalid patches
                continue

            label = 'cloudy' if (patch_mask == 2).any() else 'clear'

            patch_dir = split_dir / label / 'patch_{}'.format(patch_ids[split])
            patch_dir.mkdir(exist_ok=True, parents=True)
            tifffile.imsave(str(patch_dir / "image.tif"), patch_x)
            tifffile.imsave(str(patch_dir / "mask.tif"), patch_mask)

            patch_ids[split] += 1

    num_cloudy = len(list((output_path / 'train').glob('cloudy/*')))
    num_clear = len(list((output_path / 'train').glob('clear/*')))
    print('Done. Class balance in train: {} cloudy, {} clear'.format(num_cloudy, num_clear))


def split_train_val_test(l8_images: List[Landsat8Image], val_ratio=1 / 10, test_ratio=1 / 10, seed=None):
    # Split images randomly so that each partition contains same number of images from each biome.
    assert val_ratio + test_ratio < 1.0

    biome_to_idxs = defaultdict(list)
    for idx, l8_image in enumerate(l8_images):
        biome_to_idxs[l8_image.biome].append(idx)

    unique_biomes = biome_to_idxs.keys()
    train_val_test = [None] * len(l8_images)

    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.random_sample(len(unique_biomes))
    else:
        seeds = None

    for biome_idx, biome in enumerate(unique_biomes):
        num_tiles = len([x for x in l8_images if x.biome == biome])
        val = ["val"] * int(val_ratio * num_tiles)
        test = ["test"] * int(test_ratio * num_tiles)
        train = ["train"] * (num_tiles - (len(val) + len(test)))
        biome_train_val_test = train + val + test

        if seeds is not None:
            random.seed(seeds[biome_idx])
        random.shuffle(biome_train_val_test)

        for local_idx, global_idx in enumerate(biome_to_idxs[biome]):
            train_val_test[global_idx] = biome_train_val_test[local_idx]

    assert len(train_val_test) == len(l8_images)
    return train_val_test


def get_landsat8_images(data_path):
    def band_name(x):
        return str(x).split("_")[-1].replace(".TIF", "")

    landsat8_data = []
    image_dirs = list(data_path.glob('*/*'))  # <biome>/<image_name>
    for image_dir in image_dirs:
        biome = image_dir.parts[-2]
        name = image_dir.parts[-1]

        bands = list(image_dir.glob("**/*.TIF"))
        bands = dict(map(lambda x: (band_name(x), x), bands))

        manual_mask = next(image_dir.glob('**/*.img'))
        qa_mask = bands['BQA']

        landsat8_data.append(
            Landsat8Image(
                name=name,
                biome=biome,
                bands_15m=[bands[x] for x in BANDS_15M],
                bands_30m=[bands[x] for x in BANDS_30M],
                manual_cloud_mask=manual_mask,
                qa_cloud_mask=qa_mask
            )
        )

    return landsat8_data


def visualize_example_rgb(image, mask=None, num_classes=3):
    if image.dtype == np.uint16:
        image = np.clip(((image / (2 ** 16 - 1)).astype(np.float32) * 2.5), 0, 1)
    if mask is not None:
        f, axes = plt.subplots(1, 2, figsize=(8, 8))
        ax = axes[0]
        ax.imshow(image)
        ax.set_title('Image')
        ax.axis('off')

        ax = axes[1]
        ax.imshow(mask, vmin=0, vmax=num_classes)
        ax.set_title('Ground Truth')
        ax.axis('off')
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
    plt.show()


def read_image(image: Landsat8Image, return_profile=False):
    bands = []
    for band_path in image.bands_30m:
        with rasterio.open(band_path) as f:
            band = f.read()
            bands.append(band)
            profile = f.profile
    x = np.concatenate(bands, axis=0)
    x = np.moveaxis(x, 0, -1)

    mask = get_ground_truth(image)

    if return_profile:
        return x, mask, profile
    else:
        return x, mask


def get_ground_truth(image: Landsat8Image):
    with rasterio.open(image.manual_cloud_mask) as f:
        mask = f.read().squeeze()
    mask[mask == 0] = 0  # none
    mask[mask == 128] = 1  # Background
    mask[np.logical_or(mask == 192, mask == 255)] = 2  # thin cloud, cloud
    mask[mask == 64] = 1  # Set cloud shadow as background
    return mask


def write_generated_masks(config):
    data_path = Path(config.data_path)
    output_path = Path(config.output_dir)
    patch_size = config.patch_size

    tifs_dir = 'outputs/FixedPointGAN_1/results/tifs'
    images = get_landsat8_images(data_path)

    with open(output_path / 'assignment.txt') as f:
        train_val_test = [x.split(',')[0] for x in f.read().splitlines()]

    patch_ids = {'train': 0, 'val': 0, 'test': 0}
    for img_idx, image in enumerate(tqdm(images, desc='Writing generated masks')):
        split = train_val_test[img_idx]
        split_dir = output_path / split
        if split != 'train':
            continue  # we use real labels for evaluation
        generated_mask = tifffile.imread('{}/{}_{}_mask.tif'.format(tifs_dir, image.biome, image.name))
        generated_mask[generated_mask == 0] = 0  # none
        generated_mask[generated_mask == 128] = 1  # Background
        generated_mask[generated_mask == 255] = 2  # cloud
        ground_truth_mask = get_ground_truth(image)

        height, width = generated_mask.shape
        patches = list(product(range(0, patch_size * (height // patch_size), patch_size),
                               range(0, patch_size * (width // patch_size), patch_size)))

        for row, col in patches:
            patch_gt_mask = ground_truth_mask[row:row + patch_size, col:col + patch_size]
            patch_gen_mask = generated_mask[row:row + patch_size, col:col + patch_size]
            if (patch_gt_mask == 0).all():  # ignore patches with only invalid pixels
                continue

            label = 'cloudy' if (patch_gt_mask == 2).any() else 'clear'

            # If the image-level label is clear, we know the patch contains no clouds. In this case, we can ignore
            # the generated mask, and set the mask as all clear, reducing false positives.
            if label == 'clear':
                patch_gen_mask[patch_gen_mask == 2] = 1

            patch_dir = split_dir / label / 'patch_{}'.format(patch_ids[split])
            patch_dir.mkdir(exist_ok=True, parents=True)
            tifffile.imsave(str(patch_dir / "generated_mask.tif"), patch_gen_mask)
            patch_ids[split] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='landsat8-biome', help='Path to downloaded dataset')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size to divide images into')
    parser.add_argument('--output_dir', type=str, default='data/L8Biome')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed used to split dataset')
    parser.add_argument('--generated_masks', type=str, default=None, help='Write GAN produced cloud masks to data dir.'
                                                                          'Dir should point to tifs produced by '
                                                                          'evaluate.py, for example '
                                                                          'outputs/FixedPointGAN_1/results/tifs')
    config = parser.parse_args()
    if config.generated_masks is not None:
        write_generated_masks(config)
    else:
        prepare_patches(config)
