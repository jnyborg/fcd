import os
import pickle
import random
import shutil
from pathlib import Path

import cv2
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Subset

from evaluate import get_metrics_dict
from models.ucam import UCAM
from models.resnet50 import ResNet50
from models.cam import GradCAM, CAM, GradCAMpp

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import get_loader, L8BiomeDataset
from metrics import compute_confusion_matrix, f1_score, accuracy, AverageMeter
import metrics


class CAMSolver(object):

    def __init__(self, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if config.dataset == 'L8Biome':
            self.train_loader = get_loader(config.l8biome_image_dir, config.batch_size, 'L8Biome', 'train',
                                           config.num_workers, config.num_channels, mask_file=None)
            self.eval_batch_size = config.batch_size if config.cam_method in ['cam', 'ucam'] else 1
            self.val_loader = get_loader(config.l8biome_image_dir, self.eval_batch_size, 'L8Biome', 'val',
                                         config.num_workers, config.num_channels, mask_file='mask.tif',
                                         only_cloudy=True, pin_memory=False)

        # Model configurations.
        self.image_size = config.image_size
        self.num_channels = config.num_channels

        # Training configurations.
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.log_step = config.log_step

        # Miscellaneous.
        self.device = torch.device(config.device)
        self.mode = config.mode
        self.config = config

        # Directories.
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        if config.cam_method == 'ucam':
            self.model = UCAM('resnet34', in_channels=config.num_channels, classes=2,
                              encoder_weights='imagenet' if config.pretrained else None)
        else:
            self.model = ResNet50(n_classes=2, in_channels=config.num_channels, pretrained=config.pretrained)

        self.model.to(self.device)

        if config.cam_method == 'gradcam':
            self.cam_method = GradCAM(self.model)
        elif config.cam_method == 'gradcampp':
            self.cam_method = GradCAMpp(self.model)
        elif config.cam_method == 'cam':
            self.cam_method = CAM(self.model)
        elif config.cam_method == 'ucam':
            self.cam_method = self.model.cam
        else:
            raise NotImplemented('unknown cam method')

        self.cam_threshold = config.cam_threshold
        self.n_epochs = config.n_epochs
        self.checkpoint_dir = Path(config.model_save_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_file = self.checkpoint_dir / 'checkpoint.pt'

        if self.mode == 'train':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0)
            self.criterion = nn.CrossEntropyLoss()
            self.build_tensorboard()
            self.print_model()
            self.example_indices = random.choices(list(range(0, len(self.val_loader))), k=50)

    def print_model(self):
        """Print out the network information."""
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print(self.model)
        print(f"Number of parameters: {num_params:,}")

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        if self.config.experiment_name is not None:
            self.tensorboard_writer = SummaryWriter(log_dir=os.path.join('runs', self.config.experiment_name))
        else:
            self.tensorboard_writer = SummaryWriter()

    def accuracy_torch(self, outputs: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            preds = outputs.argmax(1)
            return preds.eq(targets).float().mean().item()

    def visualize_input_data(self):
        """Visualize input data for debugging."""
        for batch, classes, masks in self.train_loader:
            _, axes = plt.subplots(nrows=2, ncols=8, figsize=(16, 4))
            axes = axes.flatten()
            for img, c, ax, mask in zip(batch, classes, axes, masks):
                if self.num_channels > 3:
                    img = img[[3, 2, 1]]
                img = np.moveaxis(self.denorm(img).numpy(), 0, -1)
                img = np.clip(2.5 * img, 0, 1)
                ax.imshow(np.hstack([img, np.stack([mask] * 3, axis=-1) / 2]))
                ax.set_title('clear' if c == 0 else 'cloudy')
                ax.axis('off')
            plt.show()

    def train(self):
        best_val_f1, epoch, step, best_val_loss = self.restore_model(self.checkpoint_file)

        step = 0
        for epoch in range(epoch, self.n_epochs):
            progress_bar = tqdm(total=len(self.train_loader) * self.batch_size, dynamic_ncols=True)
            progress_bar.set_description('Epoch (CAM) {}'.format(epoch))
            current_lr = self.optimizer.param_groups[0]['lr']
            self.tensorboard_writer.add_scalar("train/lr", current_lr, epoch)

            # Train one epoch.
            self.model.train()
            for sample in self.train_loader:
                inputs = sample['image'].cuda(non_blocking=True, device=self.device)
                targets = sample['label'].cuda(non_blocking=True, device=self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # metrics
                if (step + 1) % self.log_step == 0:
                    batch_loss = loss.item()
                    batch_acc = self.accuracy_torch(outputs, targets)
                    progress_bar.set_postfix(loss=f'{batch_loss:.3f}', acc=f'{batch_acc:.2%}', lr=f'{current_lr:.2E}')
                    self.tensorboard_writer.add_scalar('train/loss', batch_loss, step)
                    self.tensorboard_writer.add_scalar('train/accuracy', batch_acc, step)
                progress_bar.update(self.batch_size)
                step += 1

            progress_bar.close()

            val_acc, val_f1, val_loss = self.validation(epoch)
            # is_best = best_val_loss > val_loss
            # if is_best:
            #     print('Validation loss improved from {:.3f} to {:.3f}'.format(best_val_loss, val_loss))
            #     best_val_loss = val_loss
            # else:
            #     print('Validation loss did not improve from {:.3f}'.format(best_val_loss))
            is_best = best_val_f1 < val_f1
            if is_best:
                print('Validation F1 improved from {:.3f} to {:.3f}'.format(best_val_f1, val_f1))
                best_val_f1 = val_f1
            else:
                print('Validation F1 did not improve from {:.3f}'.format(best_val_f1))

            self.scheduler.step()

            self.save_checkpoint({
                'epoch': epoch + 1,
                'step': step + 1,
                'model': self.model.state_dict(),
                'best_val_f1': best_val_f1,
                'best_val_loss': best_val_loss,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, is_best)

        self.tensorboard_writer.close()
        print('Finished training')

    def restore_model(self, model_path=None):
        if model_path is not None and model_path.exists():
            checkpoint = torch.load(str(model_path))
            epoch = checkpoint['epoch']
            step = checkpoint['step']
            best_val_f1 = checkpoint['best_val_f1'] if 'best_val_f1' in checkpoint.keys() else 0.0
            best_val_loss = checkpoint['best_val_loss'] if 'best_val_loss' in checkpoint.keys() else np.inf
            self.model.load_state_dict(checkpoint['model'])
            if self.mode == 'train':
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            print('Restored checkpoint {}, epoch {}, step {:,}, best_val_f1 {}'.format(model_path, epoch, step,
                                                                                       best_val_f1))
        else:
            epoch = 0
            step = 0
            best_val_f1 = 0.0
            best_val_loss = np.inf
        return best_val_f1, epoch, step, best_val_loss

    def save_checkpoint(self, state, is_best):
        file_path = str(self.checkpoint_dir / 'checkpoint.pt')
        torch.save(state, file_path)
        print('Saved checkpoint to {}'.format(file_path))
        if is_best:
            best_file_path = str(self.checkpoint_dir / 'best.pt')
            shutil.copyfile(file_path, best_file_path)
            print('Saved new best checkpoint to {}'.format(best_file_path))

    def print_network(self, model):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(f"Number of parameters for model: {num_params:,}")

    def validation(self, epoch):
        self.model.eval()

        cm = np.zeros((2, 2))

        losses = AverageMeter('loss', fmt=':.3f')

        for i, sample in enumerate(tqdm(self.val_loader, 'Validation')):
            inputs = sample['image'].cuda(device=self.device, non_blocking=True)
            labels = sample['label'].cuda(device=self.device, non_blocking=True)
            targets = sample['mask']

            cam, logits = self.cam_method(inputs, class_idx=1)
            cam = cam.squeeze(1)
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), inputs.size(0))

            predictions = (cam > self.cam_threshold).int().cpu().numpy()

            valid_mask = targets > 0
            y_true = targets[valid_mask] - 1
            y_pred = predictions[valid_mask]
            cm += compute_confusion_matrix(y_pred, y_true, num_classes=2)

            if i in self.example_indices:
                self.write_example_images(inputs[0], targets[0], predictions[0], cam[0], i, epoch)

        val_acc, val_f1, val_loss = accuracy(cm), f1_score(cm), losses.avg
        print('Validation Result: Accuracy={:.2%}, F1={:.4}, Classification Loss={:.4}'.format(val_acc, val_f1, losses.avg))
        self.tensorboard_writer.add_scalar('val/accuracy', val_acc, epoch)
        self.tensorboard_writer.add_scalar('val/f1', val_f1, epoch)
        self.tensorboard_writer.add_scalar('val/loss', losses.avg, epoch)

        return val_acc, val_f1, val_loss

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def write_example_images(self, image, mask, prediction, cam, example_num, step):
        image = self.denorm(image.cpu()).numpy()
        image = (image[[3, 2, 1]] * 255).astype(np.uint8)
        image = np.moveaxis(image, 0, -1)

        f, axes = plt.subplots(1, 4, figsize=(12, 2.5))
        ax = axes[0]
        ax.imshow(image)
        ax.set_title('Image')
        ax.axis('off')

        ax = axes[1]
        mask = mask.cpu().numpy()
        ax.imshow(mask, vmin=0, vmax=2)
        ax.set_title('Ground Truth')
        ax.axis('off')

        ax = axes[2]
        prediction = prediction + 1
        prediction[mask == 0] = 0
        ax.imshow(prediction, vmin=0, vmax=2)
        ax.set_title('Prediction')
        ax.axis('off')

        heatmap = cv2.applyColorMap(np.uint8(255 * cam.cpu().squeeze()), cv2.COLORMAP_JET)
        heatmap = heatmap[:, :, [2, 1, 0]]  # bgr to rgb
        result = (heatmap * 0.3 + image * 0.5).astype(np.uint8)
        ax = axes[3]
        ax.imshow(result)
        ax.set_title(self.config.cam_method)
        ax.axis('off')

        self.tensorboard_writer.add_figure('examples/image_{}/'.format(example_num), f, step)

    def find_best_threshold(self, seed=42, n_examples=10000, n_thresholds=30):
        config = self.config
        transform = Compose([
            Normalize(mean=(0.5,) * config.num_channels, std=(0.5,) * config.num_channels, max_pixel_value=2 ** 16 - 1),
            ToTensorV2(),
        ])
        dataset = L8BiomeDataset(root=config.l8biome_image_dir, transform=transform, mode='train', only_cloudy=True)
        random.seed(seed)
        indices = np.random.choice(len(dataset), n_examples, replace=False)
        dataset.images = [img for i, img in enumerate(dataset.images) if i in indices]

        batch_size = config.batch_size if config.cam_method == 'cam' else 1
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=config.num_workers, pin_memory=False)

        all_cams, all_targets = [], []
        for i, sample in enumerate(tqdm(data_loader, 'Finding best threshold for train dataset')):
            inputs = sample['image'].cuda(self.device)
            cams, logits = self.cam_method(inputs, class_idx=1)
            cams = cams.squeeze(1).cpu().numpy().astype(np.float32)

            targets = sample['mask'].numpy()
            valid_mask = targets > 0
            all_cams.append(cams[valid_mask])
            all_targets.append(targets[valid_mask] - 1)

        all_cams, all_targets = np.concatenate(all_cams), np.concatenate(all_targets)
        thresholds = np.linspace(start=all_cams.min(), stop=all_cams.max(), num=n_thresholds)

        best_f1, best_threshold = None, None
        for threshold in thresholds:
            preds = (all_cams > threshold).astype(np.uint8)
            cm = compute_confusion_matrix(preds, all_targets, 2)
            f1 = f1_score(cm)
            print('For Threshold={:.4}: F1={:.4}'.format(threshold, f1))
            if best_f1 is None or f1 >= best_f1:
                best_f1, best_threshold = f1, threshold
            else:
                break

        return best_threshold

    def eval_cam(self):
        config = self.config
        self.restore_model(Path(config.model_save_dir) / 'best.pt')
        self.model.eval()

        cam_out_dir = os.path.join(config.result_dir, self.config.cam_method)
        if not os.path.exists(cam_out_dir):
            os.mkdir(cam_out_dir)

        best_threshold = self.find_best_threshold(seed=42, n_examples=10000)

        transform = Compose([
            Normalize(mean=(0.5,) * config.num_channels, std=(0.5,) * config.num_channels, max_pixel_value=2 ** 16 - 1),
            ToTensorV2(),
        ])
        dataset = L8BiomeDataset(root=config.l8biome_image_dir, transform=transform, mode='train', only_cloudy=True)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                                  num_workers=config.num_workers, pin_memory=False)

        cm = np.zeros((2, 2))
        for i, sample in enumerate(tqdm(data_loader, 'Making CAM for train dataset')):
            inputs = sample['image'].cuda(self.device)
            cams, logits = self.cam_method(inputs, class_idx=1)
            cams = cams.squeeze(1).cpu().numpy().astype(np.float32)

            pseudo_masks = (cams > best_threshold).astype(np.uint8)
            patch_names = sample['patch_name']

            # Compute confusion matrix
            targets = sample['mask'].numpy()
            valid_mask = targets > 0
            y_true = targets[valid_mask] - 1
            y_pred = pseudo_masks[valid_mask]
            cm += compute_confusion_matrix(y_pred, y_true, num_classes=2)

            # Write pseudo masks
            # for pseudo_mask, patch_name in zip(pseudo_masks, patch_names):
            #     tifffile.imwrite(os.path.join(cam_out_dir, f'{patch_name}.tiff'), pseudo_mask)

        metrics_dict = get_metrics_dict(cm)
        pickle.dump(metrics_dict, open(os.path.join(config.result_dir, 'biome_metrics.pkl'), 'wb'))
        accuracy = metrics.accuracy(cm)
        precisions, recalls, f1_scores, supports = metrics.precision_recall_fscore_support(cm)
        print(precisions, recalls, f1_scores, supports)
        iou = metrics.iou_score(cm, reduce_mean=False)
        print('iou', iou)
        print('Overall Result: Accuracy={:.2%}, F1={:.4}, mIoU={:.4}'.format(accuracy, np.mean(f1_scores), np.mean(iou)))

