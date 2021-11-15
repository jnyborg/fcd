import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import rasterio
import sklearn.metrics as sk_metrics
import torch
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from tifffile import tifffile
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import PatchDataset
from metrics import compute_confusion_matrix, accuracy, precision_recall_fscore_support, iou_score
from prepare_landsat8_biome import get_landsat8_images, read_image, get_ground_truth

split = 'test'


# -------------------- Fixed-Point GAN ----------------------
def test_landsat8_biome(solver, config, split='test'):
    l8_images = get_l8_images(config, split=split)

    solver.restore_model(config.test_iters, only_g=True)
    # When creating generated masks for the training dataset for FCD+, we find the threshold on the training dataset
    # to avoid leaking information from the validation set into the generated labels.
    # Without pixel-wise ground truth, this could be done similarly by manual inspection.
    # best_threshold = solver.find_best_threshold('train' if split == 'train' else 'val')
    best_threshold = 0.02

    print('Evaluating model on split={} using threshold={:.4}'.format(split, best_threshold))

    overall_confusion_matrix = np.zeros((2, 2))
    all_metrics = []
    biome_wise_cm = {x.biome: np.zeros((2, 2)) for x in l8_images}

    for l8_image in tqdm(l8_images, 'Testing'):
        # Read scene to memory
        full_image, target, profile = read_image(l8_image, return_profile=True)
        valid_pixels_mask = target > 0
        target[valid_pixels_mask] -= 1

        # Compute difference and prediction using trained model
        difference = predict_scene(full_image, config, solver)
        difference[np.logical_not(valid_pixels_mask)] = 0
        mask = solver.binarize(difference, threshold=best_threshold)

        # Write difference and prediction to file
        write_tifs(difference, mask, valid_pixels_mask, profile, config, l8_image)
        write_thumbnails(full_image, mask, target, config, l8_image, thumb_size=full_image.shape[:2])  # (1000, 1000))

        # Ignore invalid pixels in metrics
        mask = mask[valid_pixels_mask].flatten()
        target = target[valid_pixels_mask].flatten()

        # Compute metrics
        cm = compute_confusion_matrix(mask, target, num_classes=2)
        metrics = get_metrics_dict(cm, l8_image.name, l8_image.biome)
        print('Result for {} ({}, {:.2%} cloud): Accuracy={:.2%}, F1={:.4} by predicting {:.2%} clouds'
              .format(l8_image.name, l8_image.biome, (target == 1).mean(), metrics['accuracy'],
                      metrics['macro avg']['f1-score'], (mask == 1).mean()))

        all_metrics.append(metrics)
        overall_confusion_matrix += cm
        biome_wise_cm[l8_image.biome] += cm

    # Store metrics to file
    metrics = get_metrics_dict(overall_confusion_matrix, name='overall', biome='all')
    print('Overall Result: Accuracy={:.2%}, F1={:.4}'.format(metrics['accuracy'], metrics['macro avg']['f1-score']))
    all_metrics.append(metrics)
    pickle.dump(all_metrics, open(os.path.join(config.result_dir, 'biome_metrics.pkl'), 'wb'))

    biome_wise_metrics = []
    for biome, cm in biome_wise_cm.items():
        biome_wise_metrics.append(get_metrics_dict(cm, name='all', biome=biome))
    pickle.dump(biome_wise_metrics, open(os.path.join(config.result_dir, 'biome_metrics_biomewise.pkl'), 'wb'))


def predict_scene(full_image, config, solver):
    patch_size = config.image_size
    batch_size = config.batch_size
    crop_size = 0
    raster_height, raster_width, _ = full_image.shape

    transforms = Compose([
        Normalize(mean=(0.5,) * config.num_channels, std=(0.5,) * config.num_channels, max_pixel_value=2 ** 16 - 1),
        ToTensorV2()
    ])

    # Cut full image into patches
    dataset = PatchDataset(full_image, patch_size, crop_size, transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                             num_workers=config.num_workers)

    # Predict each patch individually and stitch together to a full image.
    difference = np.zeros((raster_height, raster_width))
    with torch.no_grad():
        for sample in data_loader:
            if (sample['image'] == -1).all():  # Don't waste time on invalid patches
                continue
            x = sample['image'].to(config.device)
            c_trg = torch.zeros(x.shape[0], 1).to(config.device)  # translate to clear

            x_fake = solver.G(x, c_trg)
            delta = torch.abs(x_fake - x) / 2  # Divide by 2 so delta goes from [0, 2] to [0, 1]
            delta = np.mean(delta.data.cpu().numpy(), axis=1)

            # Optionally crop generated image to avoid border artifacts
            if crop_size > 0:
                delta = delta[:, crop_size:-crop_size, crop_size:-crop_size]

            rows, cols = sample['row'], sample['col']
            for batch_idx in range(len(x)):
                row, col = rows[batch_idx], cols[batch_idx]
                height = min(raster_height - row, dataset.cropped_patch_size)
                width = min(raster_width - col, dataset.cropped_patch_size)
                difference[row:row + height, col:col + width] += delta[batch_idx, :height, :width]
    return difference


# -------------------- Supervised ----------------------
def test_landsat8_biome_supervised(solver, config):
    l8_images = get_l8_images(config, split='test')

    solver.restore_model(Path(config.model_save_dir) / 'best.pt')

    overall_confusion_matrix = np.zeros((2, 2))
    all_metrics = []
    biome_wise_cm = {x.biome: np.zeros((2, 2)) for x in l8_images}
    for l8_image in tqdm(l8_images, 'Testing Supervised'):
        # Read scene to memory
        full_image, target, profile = read_image(l8_image, return_profile=True)
        valid_pixels_mask = target > 0
        target[valid_pixels_mask] -= 1

        # Compute difference and prediction using trained model
        mask = predict_scene_supervised(full_image, config, solver)

        mask[np.logical_not(valid_pixels_mask)] = 0
        write_thumbnails(full_image, mask, target, config, l8_image, thumb_size=full_image.shape[:2])

        # Ignore invalid pixels in metrics
        mask = mask[valid_pixels_mask].flatten()
        target = target[valid_pixels_mask].flatten()

        # Compute metrics
        cm = compute_confusion_matrix(mask, target, num_classes=2)
        biome_wise_cm[l8_image.biome] += cm
        metrics = get_metrics_dict(cm, l8_image.name, l8_image.biome)
        print('Result for {} ({}, {:.2%} cloud): Accuracy={:.2%}, F1={:.4} by predicting {:.2%} clouds'
              .format(l8_image.name, l8_image.biome, (target == 1).mean(), metrics['accuracy'],
                      metrics['macro avg']['f1-score'], (mask == 1).mean()))

        all_metrics.append(metrics)
        overall_confusion_matrix += cm

    # Compute overall metrics
    metrics = get_metrics_dict(overall_confusion_matrix, name='overall', biome='all')
    print('Overall Result: Accuracy={:.2%}, F1={:.4}'.format(metrics['accuracy'], metrics['macro avg']['f1-score']))
    all_metrics.append(metrics)
    pickle.dump(all_metrics, open(os.path.join(config.result_dir, 'biome_metrics.pkl'), 'wb'))

    biome_wise_metrics = []
    for biome, cm in biome_wise_cm.items():
        biome_wise_metrics.append(get_metrics_dict(cm, name='all', biome=biome))
    pickle.dump(biome_wise_metrics, open(os.path.join(config.result_dir, 'biome_metrics_biomewise.pkl'), 'wb'))


def predict_scene_supervised(full_image, config, solver):
    patch_size = config.image_size
    batch_size = config.batch_size
    crop_size = 0
    raster_height, raster_width, _ = full_image.shape

    transforms = Compose([
        Normalize(mean=(0.5,) * config.num_channels, std=(0.5,) * config.num_channels, max_pixel_value=2 ** 16 - 1),
        ToTensorV2()
    ])

    # Cut full image into patches
    dataset = PatchDataset(full_image, patch_size, crop_size, transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                             num_workers=config.num_workers)

    # Predict each patch individually and stitch together to a full image.
    predictions = np.zeros((raster_height, raster_width))
    solver.model.eval()
    with torch.no_grad():
        for sample in data_loader:
            if (sample['image'] == -1).all():  # Don't waste time on invalid patches
                continue
            x = sample['image'].to(config.device)
            output, _ = solver.model(x)
            prediction = output.argmax(1).cpu().numpy()

            # Optionally crop generated image to avoid border artifacts
            if crop_size > 0:
                prediction = prediction[:, crop_size:-crop_size, crop_size:-crop_size]

            rows, cols = sample['row'], sample['col']
            for batch_idx in range(len(x)):
                row, col = rows[batch_idx], cols[batch_idx]
                height = min(raster_height - row, dataset.cropped_patch_size)
                width = min(raster_width - col, dataset.cropped_patch_size)
                predictions[row:row + height, col:col + width] += prediction[batch_idx, :height, :width]
    return predictions


# -------------------- U-CAM ----------------------
def test_landsat8_biome_cam(solver, config):
    l8_images = get_l8_images(config, split='test')

    solver.restore_model(Path(config.model_save_dir) / 'best.pt')
    best_val_threshold = solver.find_best_threshold()
    print('Best threshold', best_val_threshold)

    overall_confusion_matrix = np.zeros((2, 2))
    all_metrics = []
    biome_wise_cm = {x.biome: np.zeros((2, 2)) for x in l8_images}
    for l8_image in tqdm(l8_images, 'Testing CAM'):
        # Read scene to memory
        full_image, target, profile = read_image(l8_image, return_profile=True)
        valid_pixels_mask = target > 0
        target[valid_pixels_mask] -= 1

        # Compute difference and prediction using trained model
        prediction = predict_scene_cam(full_image, config, solver)
        mask = (prediction > best_val_threshold).astype(np.uint8)

        mask[np.logical_not(valid_pixels_mask)] = 0
        write_thumbnails(full_image, mask, target, config, l8_image, thumb_size=(1000, 1000))

        # Ignore invalid pixels in metrics
        mask = mask[valid_pixels_mask].flatten()
        target = target[valid_pixels_mask].flatten()

        # Compute metrics
        cm = compute_confusion_matrix(mask, target, num_classes=2)
        biome_wise_cm[l8_image.biome] += cm
        metrics = get_metrics_dict(cm, l8_image.name, l8_image.biome)
        print('Result for {} ({}, {:.2%} cloud): Accuracy={:.2%}, F1={:.4} by predicting {:.2%} clouds'
              .format(l8_image.name, l8_image.biome, (target == 1).mean(), metrics['accuracy'],
                      metrics['macro avg']['f1-score'], (mask == 1).mean()))

        all_metrics.append(metrics)
        overall_confusion_matrix += cm

    # Store metrics to file
    metrics = get_metrics_dict(overall_confusion_matrix, name='overall', biome='all')
    print('Overall Result: Accuracy={:.2%}, F1={:.4}'.format(metrics['accuracy'], metrics['macro avg']['f1-score']))
    all_metrics.append(metrics)
    pickle.dump(all_metrics, open(os.path.join(config.result_dir, 'biome_metrics.pkl'), 'wb'))

    biome_wise_metrics = []
    for biome, cm in biome_wise_cm.items():
        biome_wise_metrics.append(get_metrics_dict(cm, name='all', biome=biome))
    pickle.dump(biome_wise_metrics, open(os.path.join(config.result_dir, 'biome_metrics_biomewise.pkl'), 'wb'))


def predict_scene_cam(full_image, config, solver, crop_size=0):
    patch_size = config.image_size
    batch_size = config.batch_size
    raster_height, raster_width, _ = full_image.shape

    transforms = Compose([
        Normalize(mean=(0.5,) * config.num_channels, std=(0.5,) * config.num_channels, max_pixel_value=2 ** 16 - 1),
        ToTensorV2()
    ])

    # Cut full image into patches
    dataset = PatchDataset(full_image, patch_size, crop_size, transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                             num_workers=config.num_workers)

    # Predict each patch individually and stitch together to a full image.
    predictions = np.zeros((raster_height, raster_width))
    solver.model.eval()
    with torch.no_grad():
        for sample in data_loader:
            if (sample['image'] == -1).all():  # Don't waste time on invalid patches
                continue
            x = sample['image'].to(config.device)
            cam, _ = solver.model.cam(x)
            prediction = cam.cpu().numpy().squeeze()

            # Optionally crop generated image to avoid border artifacts
            if crop_size > 0:
                prediction = prediction[:, crop_size:-crop_size, crop_size:-crop_size]

            rows, cols = sample['row'], sample['col']
            for batch_idx in range(len(x)):
                row, col = rows[batch_idx], cols[batch_idx]
                height = min(raster_height - row, dataset.cropped_patch_size)
                width = min(raster_width - col, dataset.cropped_patch_size)
                predictions[row:row + height, col:col + width] += prediction[batch_idx, :height, :width]
    return predictions


# -------------------- CFMask ----------------------
def test_landsat8_biome_fmask(config):
    l8_images = get_l8_images(config, split='test')

    overall_confusion_matrix = np.zeros((2, 2))
    all_metrics = []
    biome_wise_cm = {x.biome: np.zeros((2, 2)) for x in l8_images}
    for l8_image in l8_images:
        orig_qa_mask = tifffile.imread(str(l8_image.qa_cloud_mask))
        # Determined from https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
        # for pre-collection QA band values
        qa_mask = np.ones_like(orig_qa_mask)  # Start from all background
        qa_mask[orig_qa_mask == 1] = 0  # invalid pixels
        qa_mask[orig_qa_mask & 0b1000000000000000 == 0b1000000000000000] = 2  # medium confidence cloud
        qa_mask[orig_qa_mask & 0b1100000000000000 == 0b1100000000000000] = 2  # high confidence cloud
        qa_mask[orig_qa_mask & 0b0010000000000000 == 0b0010000000000000] = 2  # medium confidence cirrus
        qa_mask[orig_qa_mask & 0b0011000000000000 == 0b0011000000000000] = 2  # high confidence cirrus

        target = get_ground_truth(l8_image)

        # Ignore invalid pixels
        valid_pixels_mask = target > 0
        target = target[valid_pixels_mask] - 1
        qa_mask = qa_mask[valid_pixels_mask] - 1

        # Compute metrics
        cm = compute_confusion_matrix(qa_mask, target, num_classes=2)
        biome_wise_cm[l8_image.biome] += cm
        metrics = get_metrics_dict(cm, l8_image.name, l8_image.biome)
        print('Result for {} ({}, {:.2%} cloud): Accuracy={:.2%}, F1={:.4} by predicting {:.2%} clouds'
              .format(l8_image.name, l8_image.biome, (target == 1).mean(), metrics['accuracy'],
                      metrics['macro avg']['f1-score'], (qa_mask == 1).mean()))

        all_metrics.append(metrics)
        overall_confusion_matrix += cm

    # Store metrics to file
    metrics = get_metrics_dict(overall_confusion_matrix, name='overall', biome='all')
    print('Overall Result: Accuracy={:.2%}, F1={:.4}'.format(metrics['accuracy'], metrics['macro avg']['f1-score']))
    all_metrics.append(metrics)
    fmask_result_dir = os.path.join('outputs', 'fmask')
    os.makedirs(fmask_result_dir, exist_ok=True)
    pickle.dump(all_metrics, open(os.path.join(fmask_result_dir, 'biome_metrics.pkl'), 'wb'))
    biome_wise_metrics = []
    for biome, cm in biome_wise_cm.items():
        biome_wise_metrics.append(get_metrics_dict(cm, name='all', biome=biome))
    pickle.dump(biome_wise_metrics, open(os.path.join(fmask_result_dir, 'biome_metrics_biomewise.pkl'), 'wb'))

# ----- Helper functions -------
# Taken from https://github.com/developmentseed/landsat-util/blob/develop/landsat/image.py
def to_uint8_based_on_cloud_coverage(bands, cloud_coverage=100.0):
    output = np.zeros_like(bands, dtype=np.uint8)
    for i, band in enumerate(bands):
        # Color Correction
        band = _color_correction(band, 0, cloud_coverage)
        band = img_as_ubyte(band)
        output[i] = band
    return output


def _color_correction(band, low, coverage):
    p_low, cloud_cut_low = _percent_cut(band, low, 100 - (coverage * 3 / 4))
    temp = np.zeros(np.shape(band), dtype=np.uint16)
    cloud_divide = 65000 - coverage * 100
    mask = np.logical_and(band < cloud_cut_low, band > 0)
    temp[mask] = rescale_intensity(band[mask], in_range=(p_low, cloud_cut_low), out_range=(256, cloud_divide))
    temp[band >= cloud_cut_low] = rescale_intensity(band[band >= cloud_cut_low],
                                                    out_range=(cloud_divide, 65535))
    return temp


def _percent_cut(color, low, high):
    return np.percentile(color[np.logical_and(color > 0, color < 65535)], (low, high))


def write_tifs(difference, prediction, valid_mask, profile, config, l8_image):
    profile.update(driver='GTiff', count=1, compress='lzw', dtype='uint8')
    output_dir = os.path.join(config.result_dir, 'tifs')
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f'{l8_image.biome}_{l8_image.name}')
    with rasterio.open('{}_difference.tif'.format(prefix), 'w', **profile) as dst:
        dst.write((difference[np.newaxis, ...] * 255).astype(np.uint8))

    mask = prediction.copy()
    mask[mask == 0] = 128
    mask[mask == 1] = 255
    mask[~valid_mask] = 0
    with rasterio.open('{}_mask.tif'.format(prefix), 'w', **profile) as dst:
        dst.write((mask[np.newaxis, ...]).astype(np.uint8))


def get_l8_images(config, split='test'):
    l8_images = get_landsat8_images(Path(config.orig_image_dir))
    with open(os.path.join(config.l8biome_image_dir, 'assignment.txt')) as f:
        lines = [x.split(',') for x in f.read().splitlines()]
        lines = [(x[-1]) for x in lines if x[0] == split]
    return [x for x in l8_images if x.name in lines]


def write_thumbnails(image, prediction, target, config, l8_image, thumb_size=(1000, 1000)):
    output_dir = os.path.join(config.result_dir, 'l8_thumbnails')
    file_prefix = f'{l8_image.biome}_{l8_image.name}'

    image = image.copy()[..., [3, 2, 1]]
    image = np.moveaxis(image, -1, 0)  # Bands first
    image = to_uint8_based_on_cloud_coverage(image, cloud_coverage=(target == 1).mean() * 100.0)
    image = np.moveaxis(image, 0, -1)  # Bands last
    thumb = cv2.resize(image, thumb_size)

    # color = (63, 255, 63)
    color = (26, 178, 255)
    # color = (255, 0, 255)
    mask = cv2.resize(prediction, thumb_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask[..., np.newaxis] * np.array(color)).astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, thumb, 0.5, 0.)
    ind = np.any(mask > 0, axis=-1)
    thumb_pred = thumb.copy()
    thumb_pred[ind] = weighted_sum[ind]

    mask = cv2.resize(target, thumb_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask[..., np.newaxis] * np.array(color)).astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, thumb, 0.5, 0.)
    ind = np.any(mask > 0, axis=-1)
    thumb_target = thumb.copy()
    thumb_target[ind] = weighted_sum[ind]

    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, file_prefix + '_image.jpg'), thumb[..., [2, 1, 0]])  # cv2 uses BGR format
    cv2.imwrite(os.path.join(output_dir, file_prefix + '_prediction.jpg'), thumb_pred[..., [2, 1, 0]])
    cv2.imwrite(os.path.join(output_dir, file_prefix + '_target.jpg'), thumb_target[..., [2, 1, 0]])


def get_metrics_dict(confusion_matrix, name=None, biome=None):
    acc = accuracy(confusion_matrix)
    p, r, f1, s = precision_recall_fscore_support(confusion_matrix)
    iou = iou_score(confusion_matrix, reduce_mean=False)
    return {
        'name': name,
        'biome': biome,
        'clear': {
            'precision': p[0],
            'recall': r[0],
            'f1-score': f1[0],
            'iou-score': iou[0],
            'support': s[0],
        },
        'cloudy': {
            'precision': p[1],
            'recall': r[1],
            'f1-score': f1[1],
            'iou-score': iou[0],
            'support': s[1],
        },
        'accuracy': acc,
        'macro avg': {
            'precision': np.mean(p),
            'recall': np.mean(r),
            'f1-score': np.mean(f1),
            'iou-score': np.mean(iou),
            'support': np.sum(s),
        }
    }
