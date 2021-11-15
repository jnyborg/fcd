# FCD: Fixed-Point GAN for Cloud Detection
PyTorch source code of [Nyborg & Assent (2020)]().

> TODO
> abstract
> and
> authors


## Dependencies
To setup a conda environment named `fcd` with all dependencies installed, run 

```
conda env create -f environment.yml
conda activate fcd
``` 

This will install the following packages:
```
tqdm
opencv-python
rasterio
tifffile
pillow
matplotlib
pytorch
torchvision
cudatoolkit
tensorboard
albumentations
sklearn
segmentation-models-pytorch
```

## Usage
To download the full Landsat-8 Biome dataset (96 Landsat-8 scenes, about 182 GB when extracted), run
```
python download_landsat8_biome.py
```

To prepare 128x128 patches with image-level labels for training, run
```
python prepare_landsat8_biome.py 
```

### Train FCD
To train Fixed-Point GAN for Cloud Detection (FCD), run
```
python main.py --mode train --dataset L8Biome --image_size 128 --batch_size 16 --experiment_name FCD
```

You can monitor the training progress by starting TensorBoard for the `runs` dir:
```
tensorboard --logdir=runs
```


### Train FCD+
When FCD is trained, we can generate pixel-level cloud masks for the training dataset by running
```
python main.py --mode generate_masks --batch_size 64 --experiment_name FCD
```
This will generate cloud masks for the Landsat-8 scenes in the training dataset, and save them in `outputs/FCD/results/tifs`. 
Then, to divide these cloud masks into the corresponding patches for training, we can run
```
python prepare_landsat8_biome.py --generated_masks outputs/FCD/results/tifs
```
resulting in a `generated_mask.tif` in addition to the ground truth `mask.tif` for every training patch.

Then, to train FCD+ with `generated_mask.tif` as targets, run
```
python supervised_main.py --mode train --batch_size 64 --train_mask_file generated_mask.tif \
                          --classifier_head True --experiment_name FCD+
```

Finally, to fine-tune the resulting model on 1% of actual pixel-wise ground truth, run
```
python supervised_main.py --mode train --batch_size 64 --keep_ratio 0.01 --lr 1e-5 --freeze_encoder True \
                          --model_weights outputs/FCDPlus/models/best.pt \
                          --experiment_name FCD+1Pct 
```

### Train models compared with in paper
See the bash scripts in the `scripts` folder for the exact runs done in the paper.


# Citation
If you find our work useful for your research, please site our [paper](TODO):
```
TODO citation info here
```



# Acknowledgements
This repository is based on [mahfuzmohammad/Fixed-Point-GAN](https://github.com/mahfuzmohammad/Fixed-Point-GAN) and [yunjey/stargan](https://github.com/yunjey/stargan).






