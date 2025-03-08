# Real-Time Anomaly Segmentation for Road Scenes

This repository contains the code for the project **Real-Time Anomaly Segmentation for Road Scenes**, conducted as part of the Advanced Machine Learning course 2024/2025 at Politecnico di Torino. The project explores the application of deep learning models for real-time per-pixel anomaly detection in road scenes.

This repository was built using the base code from [AnomalySegmentation_CourseProjectBaseCode](https://github.com/shyam671/AnomalySegmentation_CourseProjectBaseCode) as a foundation. Enhancements and modifications were made to improve the functionality and adapt it to specific project requirements.

For more details, refer to the [full paper](https://github.com/RonPlusSign/AnomalySegmentation/blob/main/s331998_s306027_s330519.pdf).

## Visual example
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/5e792dd3-9557-411f-ad22-549c941375f8" width="700">
</div>

This visual comparison showcases the anomaly segmentation results using **MSP** and **MaxLogit**, applied to an image from the Road Anomaly dataset. 

- **Top Left**: Original input image from the dataset.  
- **Top Right**: Ground truth segmentation.  
- **Bottom Left**: Anomaly segmentation output using **MSP**.  
- **Bottom Right**: Anomaly segmentation output using **MaxLogit**.

## Features

- Training and evaluation scripts for semantic segmentation models such as **ENet**, **ERFNet**, and **BiSeNet**.
- Methods for anomaly segmentation using various techniques, including Maximum Softmax Probability (MSP), MaxLogit, and Mahalanobis Distance.
- Incorporates enhancements like **temperature scaling**, **void classifiers**, and advanced loss functions and OoD methods (e.g., IsoMax+ and LogitNorm).
- Benchmarked on datasets such as **Cityscapes**, **RoadAnomaly**, **Fishyscapes**, and **SegmentMeIfYouCan**.

## Repository Structure

- **[train](train/)**: Tools and scripts for training models on the Cityscapes dataset.
- **[eval](eval/)**: Scripts for evaluating models, visualizing results, and performing anomaly segmentation.
- **[trained_models](trained_models/)**: Contains pretrained models used in the project.
- **[save](save/)**: Contains the fine-tuned models trained during this project. These models have been optimized for anomaly segmentation tasks based on the experiments conducted.
  - **Void Classification**: BiSeNet, ENet and ERFNet trained on 20 classes, 19 + void, on Cityscape dataset (`bisenet_training_void`, `enet_training_void` `erfnet_training_void`)
  - **ERFNet with Loss Variants**:: ERFNet trained with different loss functions and OoD methods on Cityscape dataset (`erfnet_training_focal_loss`, `erfnet_training_isomaxplus_cross_entropy_loss`, `erfnet_training_isomaxplus_focal_loss`, `erfnet_training_logitnorm_cross_entropy_loss`, `erfnet_training_logitnorm_focal_loss`).
-  **[Project6.ipynb](Project6.ipynb/)**: Jupyter Notebook compatible with Google Colab to run all scripts for training, validation, and visualization of models.

## Requirements

1. **Datasets**:
   - [Cityscapes dataset](https://www.cityscapes-dataset.com/): Download "leftImg8bit" for RGB images and "gtFine" for labels. Convert labels to `_labelTrainIds` using the [Cityscapes conversion script](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py).
   - Testing datasets:
     - [Segment Me If You Can (RoaDAnomaly21, RoadObstacle21)](https://segmentmeifyoucan.com/datasets)
     - [Fishyscapes (FS Static, FS Lost&Found)](https://fishyscapes.com/dataset)
     - [Road Anomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/)
2. **Environment**:
   - Python 3.6
   - [PyTorch](https://pytorch.org/) with CUDA (tested with CUDA 8.0).
   - Additional Python packages: `numpy`, `matplotlib`, `Pillow`, `torchvision`, `visdom` (optional for visualization).
3. Clone this repository

## How to Use

### Training

Train the model on the Cityscapes dataset:
```bash
python train/main.py \
    --savedir <savedir> \
    --datadir <data_dir> \
    --model <model_name> \
    --cuda \
    --num-epochs 20 \
    [--FineTune] \
    [--loadWeights <pretrained_weight_path>] \
    [--void] \
    [--loss <loss_name>] \
    [--logit_normalization]
```

Replace:
- `<savedir>`: Directory to save the trained model.
- `<data_dir>`: Path to the Cityscapes dataset.
- `<model_name>`: Name of the model (`erfnet`, `enet`, `bisenet`  or `erfnet_isomaxplus` for IsoMax+).
- `[--FineTune]`: Optional flag to enable fine-tuning.
- `[--loadWeights <pretrained_weight_path>]`: Optional; specify the path to the pretrained weights for fine-tuning.
- `[--void]`: Optional flag to enable void classification for anomaly detection.
- `[--loss <loss_name>]`: Optional; specify the loss function for fine-tuning. Options are:
  - `CrossEntropy`: Default cross-entropy loss.
  - `FocalLoss`: Focal loss for harder examples.
- `[--logit_normalization]`: Optional flag to enable logit normalization during training.

For **IsoMax+**, set the `--model` parameter to `erfnet_isomaxplus` to enable this method.

### Anomaly Segmentation and Visualization

Run inference for anomaly segmentation and optionally visualize the results:

```bash
python eval/evalAnomaly.py \
    --input '/content/Validation_Dataset/<dataset_dir>/images/*.png' \
    --method <method> \
    --model <model_name> \
    --loadDir <load_dir> \
    --loadWeights <weight_path> \
    [--save-plots-dir <save_plots_dir>] \
    [--save-colored <save_image_path>]
```

Replace:
- `<dataset_dir>`: Directory containing the dataset images.
- `<method>`: Anomaly detection method (`MSP`, `MaxLogit`, `MaxEntropy` or `Mahalanobis`).
- `<model_name>`: Name of the model used.
- `<load_dir>`: Directory of the trained model.
- `<weight_path>`: Path to the model weights.
- `[--save-plots-dir <save_plots_dir>]`: Optional; Directory to save the generated anomaly plots.
- `[--save-colored <save_image_path>]`: Optional; specify this parameter to save colorized segmentation results.

This command performs anomaly segmentation on the specified dataset. Adding the `--save-colored` option will save visually enhanced results for easier interpretation.

## Results

The project evaluated multiple models and methods on key metrics such as:
- **AuPRC (Area under Precision-Recall Curve)**
- **FPR95 (False Positive Rate at 95% True Positive Rate)**
- **mIoU (Mean Intersection over Union)**

Key insights:
- **MaxLogit** consistently outperformed other anomaly detection methods.
- Fine-tuning models with the void classifier and advanced loss functions enhanced performance on out-of-distribution (OoD) detection.
- **BiSeNet** demonstrated the best real-time performance with high frames per second (FPS).


## Citation

If you use this repository, please cite our work:
```
@inproceedings{delli_modi_dellisanti2025anomalysegmentation,
  title={Real-Time Anomaly Segmentation for Road Scenes},
  author={Delli, Andrea and Dellisanti, Christian and Modi, Giorgia},
  year={2025},
  institution={Politecnico di Torino}
}
```
