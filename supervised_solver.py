import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import get_loader
from metrics import compute_confusion_matrix, f1_score, accuracy


class SupervisedSolver(object):

    def __init__(self, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if config.dataset == 'L8Biome':
            self.train_loader = get_loader(config.l8biome_image_dir, config.batch_size, 'L8Biome', 'train',
                                           config.num_workers, config.num_channels, mask_file=config.train_mask_file,
                                           keep_ratio=config.keep_ratio)
            self.val_loader = get_loader(config.l8biome_image_dir, config.batch_size, 'L8Biome', 'val',
                                         config.num_workers, config.num_channels, mask_file='mask.tif')

        # Model configurations.
        self.image_size = config.image_size
        self.num_channels = config.num_channels

        # Training configurations.
        self.batch_size = config.batch_size
        self.lr = config.lr

        # Miscellaneous.
        self.device = torch.device(config.device)
        self.mode = config.mode
        self.config = config

        # Directories.
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        self.train_mask_file = config.train_mask_file
        self.keep_ratio = config.keep_ratio
        self.encoder_weights = config.encoder_weights
        self.model_weights = config.model_weights
        self.train_encoder_only = config.train_encoder_only
        self.log_step = config.log_step
        self.classifier_head = config.classifier_head

        classification_head_params = {'classes': 1, 'pooling': "avg", 'dropout': 0.2, 'activation': None}
        if self.encoder_weights in [None, 'imagenet']:
            self.model = smp.Unet('resnet34', in_channels=self.num_channels, classes=2,
                                  encoder_weights=self.encoder_weights, aux_params=classification_head_params)
        else:
            # Load encoder weights from file
            self.model = smp.Unet('resnet34', in_channels=self.num_channels, classes=2, encoder_weights=None, aux_params=classification_head_params)
            if not os.path.exists(self.encoder_weights):
                raise FileNotFoundError('Encoder weights path {} did not exist, exiting.'.format(self.encoder_weights))
            encoder_weights = torch.load(self.encoder_weights)
            self.model.encoder.load_state_dict(encoder_weights)
            print('Loaded encoder weights from', self.encoder_weights)

        if self.model_weights is not None:
            if not os.path.exists(self.model_weights):
                raise FileNotFoundError('Model weights path {} did not exist, exiting.'.format(self.model_weights))
            state = torch.load(self.model_weights)
            self.model.load_state_dict(state['model'])
            print('Initialized model with weights from {}'.format(self.model_weights))

        if config.freeze_encoder:
            print('Freezing encoder weights')
            self.model.encoder.requires_grad_(False)

        # self.visualize_input_data()

        if self.mode == 'train':
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=3, factor=0.1, verbose=True, mode='max')
            self.build_tensorboard()
            self.print_model()
            example_indices = list(range(0, len(self.val_loader)))
            self.example_indices = random.choices(example_indices, k=50)


        self.model.to(self.device)
        self.n_epochs = config.n_epochs
        self.checkpoint_dir = Path(config.model_save_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_file = self.checkpoint_dir / 'checkpoint.pt'

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
            preds = outputs.argmax(dim=1)
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
        if self.train_encoder_only:
            self.train_classifier()
            return

        best_val_f1, epoch, step = self.restore_model(self.checkpoint_file)
        ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # use CE loss instead of BCE so we can ignore unknown class
        bce_criterion = nn.BCEWithLogitsLoss()

        step = 0
        for epoch in range(epoch, self.n_epochs):
            self.model.train()
            tq = tqdm(total=len(self.train_loader) * self.batch_size, dynamic_ncols=True)
            tq.set_description('Epoch {}'.format(epoch))

            # Train one epoch.
            for inputs, target_labels, target_masks in self.train_loader:
                inputs = self.cuda(inputs)
                target_masks = self.cuda(target_masks - 1)  # set invalid as -1, and ignore in loss.
                target_labels = self.cuda(target_labels)

                masks, labels = self.model(inputs)
                if self.classifier_head:
                    loss = ce_criterion(masks, target_masks) + bce_criterion(labels, target_labels)
                else:
                    loss = ce_criterion(masks, target_masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # metrics
                if (step + 1) % self.log_step == 0:
                    batch_loss = loss.item()
                    batch_acc = self.accuracy_torch(masks, target_masks)
                    tq.set_postfix(loss='{:.3f}'.format(batch_loss), acc='{:.2%}'.format(batch_acc))
                    self.tensorboard_writer.add_scalar('supervised/train_loss', batch_loss, step)
                    self.tensorboard_writer.add_scalar('supervised/train_accuracy', batch_acc, step)
                tq.update(self.batch_size)
                step += 1

            tq.close()

            val_loss, val_acc, val_f1 = self.validation(epoch)
            is_best = best_val_f1 < val_f1
            if is_best:
                print('Validation F1 improved from {:.3f} to {:.3f}'.format(best_val_f1, val_f1))
                best_val_f1 = val_f1
            else:
                print('Validation F1 did not improve from {:.3f}'.format(best_val_f1))

            self.scheduler.step(val_f1)

            self.save_checkpoint({
                'epoch': epoch + 1,
                'step': step + 1,
                'model': self.model.state_dict(),
                'best_val_f1': best_val_f1,
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
            self.model.load_state_dict(checkpoint['model'])
            if self.mode == 'train':
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            print('Restored checkpoint {}, epoch {}, step {:,}, best_val_f1 {}'.format(model_path, epoch, step, best_val_f1))
        else:
            epoch = 0
            step = 0
            best_val_f1 = 0.0
        return best_val_f1, epoch, step

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

        losses = []
        cm = np.zeros((2, 2))
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        with torch.no_grad():
            for i, (inputs, _, targets) in enumerate(tqdm(self.val_loader, 'Validation')):
                outputs, _ = self.model(self.cuda(inputs))
                outputs = outputs.cpu()
                targets = targets - 1
                loss = criterion(outputs, targets).numpy()
                losses.append(loss)

                valid_mask = targets > -1
                predictions = outputs.numpy().argmax(axis=1)
                targets = targets.numpy()
                cm += compute_confusion_matrix(predictions[valid_mask], targets[valid_mask], 2)

        losses = np.concatenate(losses)
        loss = np.mean(losses)
        acc, f1 = accuracy(cm), f1_score(cm)
        print('Validation Result: Loss={:.4}, Accuracy={:.2%}, F1={:.4}'.format(loss, acc, f1))
        self.tensorboard_writer.add_scalar('supervised/val_loss', loss, epoch)
        self.tensorboard_writer.add_scalar('supervised/val_acc', acc, epoch)
        self.tensorboard_writer.add_scalar('supervised/val_f1', f1, epoch)

        return loss, acc, f1

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def cuda(self, x):
        return x.cuda(device=self.device, non_blocking=True) if torch.cuda.is_available() else x

    def train_classifier(self):
        """
        Train only the encoder part of U-Net, for pretraining on image-level dataset.
        """
        best_val_f1, epoch, step = 0, 0, 0
        criterion = nn.BCEWithLogitsLoss()

        # Ensure that we don't train the decoder.
        self.model.decoder.requires_grad_(False)
        self.model.segmentation_head.requires_grad_(False)

        for epoch in range(epoch, self.n_epochs):
            tq = tqdm(total=len(self.train_loader) * self.batch_size, dynamic_ncols=True)
            tq.set_description('Epoch {} (train_classifier)'.format(epoch))

            # Train one epoch.
            self.model.train()
            for inputs, targets in self.train_loader:
                inputs = self.cuda(inputs)
                targets = self.cuda(targets)  # set invalid as -1, and ignore in loss.

                _, outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # metrics
                if (step + 1) % self.log_step == 0:
                    batch_loss = loss.item()
                    with torch.no_grad():
                        preds = (outputs > 0.5).float()
                        batch_acc = preds.eq(targets).float().mean().item()
                    tq.set_postfix(loss='{:.3f}'.format(batch_loss), acc='{:.2%}'.format(batch_acc))
                    self.tensorboard_writer.add_scalar('supervised/train_loss', batch_loss, step)
                    self.tensorboard_writer.add_scalar('supervised/train_accuracy', batch_acc, step)
                tq.update(self.batch_size)
                step += 1

            tq.close()

            # Run validation
            val_losses = []
            val_cm = np.zeros((2, 2))
            self.model.eval()
            for i, (inputs, targets, _) in enumerate(tqdm(self.val_loader, 'Validation')):
                with torch.no_grad():
                    _, outputs = self.model(self.cuda(inputs))
                outputs = outputs.cpu()
                predictions = (outputs > 0.5).float()
                val_losses.append(criterion(outputs, targets).numpy())
                val_cm += compute_confusion_matrix(predictions, targets.numpy(), 2)

            val_loss, val_acc, val_f1 = np.mean(val_losses), accuracy(val_cm), f1_score(val_cm)
            print('Validation Result: Loss={:.4}, Accuracy={:.2%}, F1={:.4}'.format(val_loss, val_acc, val_f1))
            self.tensorboard_writer.add_scalar('supervised/val_loss', val_loss, epoch)
            self.tensorboard_writer.add_scalar('supervised/val_acc', val_acc, epoch)
            self.tensorboard_writer.add_scalar('supervised/val_f1', val_f1, epoch)
            is_best = best_val_f1 < val_f1
            if is_best:
                print('Validation F1 improved from {:.3f} to {:.3f}'.format(best_val_f1, val_f1))
                best_val_f1 = val_f1
            else:
                print('Validation F1 did not improve from {:.3f}'.format(best_val_f1))

            self.scheduler.step(val_f1)

            resnet34_state_dict = self.model.encoder.state_dict()
            # Add these keys for compatiability with torchvision resnet34
            resnet34_state_dict['fc.bias'] = None
            resnet34_state_dict['fc.weight'] = None

            if is_best:
                file_path = str(self.checkpoint_dir / 'l8biome_resnet34.pt')
                torch.save(resnet34_state_dict, file_path)
                print('Saved encoder weights to {}'.format(file_path))

        self.tensorboard_writer.close()
        print('Finished training')


    def visualize_preds(self):
        from pathlib import Path
        from torchvision.utils import save_image
        from albumentations import Normalize, Compose
        from albumentations.pytorch.transforms import ToTensorV2
        from data_loader import L8BiomeDataset
        from torch.utils import data
        config = self.config

        class ConcatDataset(torch.utils.data.Dataset):
            def __init__(self, *datasets):
                self.datasets = datasets

            def __getitem__(self, i):
                return tuple(d[i] for d in self.datasets)

            def __len__(self):
                return min(len(d) for d in self.datasets)

        transform = []
        transform.append(Normalize(mean=(0.5,) * config.num_channels, std=(0.5,) * config.num_channels, max_pixel_value=2 ** 16 - 1))
        transform.append(ToTensorV2())
        transform = Compose(transform)

        dataset_gen = L8BiomeDataset(config.l8biome_image_dir, transform, 'train', 'generated_mask.tif')
        dataset_man = L8BiomeDataset(config.l8biome_image_dir, transform, 'train', 'mask.tif')

        dataset = ConcatDataset(dataset_gen, dataset_man)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        output_dir = Path('example_fcdplus')
        output_dir.mkdir(exist_ok=True)

        def to_rgb(tensor):
            return (3.5 * self.denorm(tensor[:, [3, 2, 1]])).clamp(0, 1)

        def to_rgb_mask(tensor):
            t = tensor.repeat(3, 1, 1) * torch.Tensor([26, 178, 255]).reshape(-1, 1, 1)
            return t / 255

        data = []
        for i in range(100):
            print(i)
            (inputs, label, mask), (_, _, gt) = next(data_iter)
            invalid_pixels_mask = gt == 0
            
            mask = mask - 1
            mask[invalid_pixels_mask] = 0

            gt = gt - 1
            gt[invalid_pixels_mask] = 0

            label = 'clear' if (label == 0).all() else 'cloudy'
            data.append((label, inputs))

            patch_output_dir = output_dir / label
            patch_output_dir.mkdir(exist_ok=True)
            save_image(to_rgb(inputs), str(patch_output_dir / f'{i}_1input.png'))
            save_image(to_rgb_mask(mask), str(patch_output_dir / f'{i}_fcd.png'))
            save_image(to_rgb_mask(gt), str(patch_output_dir / f'{i}_gt.png'))

        best_val_f1, epoch, step = self.restore_model(Path('outputs/FullySupervised_Generated_1/models') / 'best.pt')
        for i, (label, inputs) in enumerate(data):
            patch_output_dir = output_dir / label
            self.model.eval()
            with torch.no_grad():
                preds, _ = self.model(inputs.cuda())

            preds = preds.argmax(dim=1).cpu()
            save_image(to_rgb_mask(preds), str(patch_output_dir / f'{i}_fcdplus.png'))

        best_val_f1, epoch, step = self.restore_model(Path('outputs/FineTune1Pct_FullySupervised_Generated_1/models') / 'best.pt')
        for i, (label, inputs) in enumerate(data):
            patch_output_dir = output_dir / label
            self.model.eval()
            with torch.no_grad():
                preds, _ = self.model(inputs.cuda())

            preds = preds.argmax(dim=1).cpu()
            save_image(to_rgb_mask(preds), str(patch_output_dir / f'{i}_fcdplus1pct.png'))



