# Clothes Segmentation — CV-CY002

A semantic segmentation system for extracting **clothing regions from images of people**.

The project uses **DeepLabV3+ with a ResNet-50 encoder** and converts the LIP human-parsing dataset into a binary segmentation task:

* `0` — Non-clothes / Background
* `1` — Clothes

The project includes dataset preprocessing, model training, evaluation, checkpointing, and inference on personal images.

---

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Model](#model)
* [Dataset](#dataset)
* [Requirements](#requirements)
* [Installation](#installation)
* [Training](#training)
* [Resume Training](#resume-training)
* [Evaluation](#evaluation)
* [Inference on a Personal Image](#inference-on-a-personal-image)
* [Reproducing the Results](#reproducing-the-results)
* [Pretrained Weights](#pretrained-weights)
* [Report](#report)
* [Notebook](#evaluation-notebook)

---

## Overview

The goal of this project is to build a clothing segmentation model that identifies clothing pixels in an input image.

The original LIP dataset contains multiple human-parsing classes. For this project, the labels are mapped into two classes:

| Class ID | Class                    |
| -------: | ------------------------ |
|      `0` | Non-clothes / Background |
|      `1` | Clothes                  |

The resulting segmentation mask can be used as a preprocessing step for applications such as **virtual fitting rooms** and other clothing-related computer vision systems.

---

## Project Structure

```text
clothes-segmentation-cvcy002/
│
├── configs/
│   └── config.yaml
│
├── notebooks/
│   ├── cycv002-train-evaluate.ipynb
│   ├── cycv002_evaluate.ipynb
│   └── cycv002_predict.ipynb
│
├── scripts/
│   ├── LIPpreprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── src/
│   └── cvcy002/
│       ├── data/
│       ├── models/
│       ├── training/
│       └── evaluation/
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Model

The project uses:

**DeepLabV3+ + ResNet-50**

### Configuration

| Parameter         | Value               |
| ----------------- | ------------------- |
| Architecture      | DeepLabV3+          |
| Encoder           | ResNet-50           |
| Encoder weights   | ImageNet pretrained |
| Number of classes | 2                   |
| Input resolution  | 512 × 512           |
| Optimizer         | AdamW               |
| Learning rate     | `1e-4`              |
| Batch size        | `16`                |
| Maximum epochs    | `30`                |

The training configuration can be found in:

```text
configs/config.yaml
```

---

## Dataset

The project uses the **Look Into Person (LIP)** human-parsing dataset.

LIP provides pixel-level annotations for human-centric images. The original multi-class annotations are converted into a binary clothing segmentation task during preprocessing.

For more information about LIP, see:

* [LIP — Look Into Person](https://sysu-hcp.net/lip/)
* [LIP CVPR 2017 Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Gong_Look_Into_Person_CVPR_2017_paper.html)
* [LIP Dataset in Kaggle](https://www.kaggle.com/datasets/roneoz/lip-dataset/data?select=Train)

### Adding the Dataset to Kaggle

The training workflow was performed using a Kaggle Notebook with GPU acceleration.

Before running the training code, the LIP dataset must be attached to the Kaggle Notebook as an Input/Data Source.

Steps
Open the LIP Dataset on Kaggle.
Open your Kaggle training notebook.
In the notebook interface, open the Input / Add Input section.
Search for the LIP dataset.
Add the dataset to the notebook.
Verify that the dataset appears under the notebook's input files.

Kaggle makes attached datasets available through the /kaggle/input/ directory.

### Dataset preprocessing

Before training, run:

```bash
MPLBACKEND=Agg uv run python scripts/LIPpreprocessing.py
```

The preprocessing step verifies and prepares the dataset and provides sample visualizations for checking the label mapping.

---

# Requirements

The project requires:

* Python `3.10+`
* Git
* `uv`
* NVIDIA GPU with CUDA support recommended for training

The exact Python version used by the project is specified in:

```text
.python-version
```

Project dependencies are defined in:

```text
pyproject.toml
```

and locked in:

```text
uv.lock
```

Using the lock file helps reproduce the project environment consistently.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/abdulrahman1238/clothes-segmentation-cvcy002.git
cd clothes-segmentation-cvcy002
```

## 2. Install uv

If `uv` is not already installed:

```bash
pip install uv
```

## 3. Create the project environment and install dependencies

```bash
uv sync
```

`uv sync` creates/synchronizes the project's environment using the project dependency configuration and lock file.

## 4. Verify the installation

You can verify that the project environment is working with:

```bash
uv run python --version
```

---

# Training

The model was trained using **Kaggle GPU**.

The following procedure reproduces the training workflow.

## 1. Install uv

In a Kaggle notebook:

```python
!pip install uv --quiet
```

## 2. Clone the repository

```python
!git clone https://github.com/abdulrahman1238/clothes-segmentation-cvcy002
```

## 3. Enter the project directory

```python
%cd /kaggle/working/clothes-segmentation-cvcy002
```

## 4. Install dependencies

```python
!uv sync
```

You should see:

```text
Setup complete!
```

## 5. Run preprocessing

Before training, verify the dataset and label preprocessing:

```python
!MPLBACKEND=Agg uv run python scripts/LIPpreprocessing.py
```

The preprocessing script also helps verify that the converted masks contain the expected classes.

## 6. Train the model

```python
!MPLBACKEND=Agg uv run python scripts/train.py --config configs/config.yaml
```

Training outputs, including checkpoints, are saved under:

```text
outputs/checkpoints/
```

The main checkpoints are:

```text
outputs/checkpoints/best_model.pth
outputs/checkpoints/last_model.pth
```

### Best model

`best_model.pth` contains the checkpoint corresponding to the best validation performance during training.

### Last model

`last_model.pth` contains the most recent training checkpoint and can be used to resume training.

---

# Resume Training

If training is interrupted, training can be resumed from the last checkpoint:

```python
!MPLBACKEND=Agg uv run python scripts/train.py \
    --resume outputs/checkpoints/last_model.pth
```

This allows the training process to continue instead of starting again from the beginning.

---

# Evaluation

After training, evaluate the best checkpoint using:

```python
!MPLBACKEND=Agg uv run python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth
```

The evaluation pipeline calculates segmentation metrics including:

* Pixel Accuracy
* IoU
* Mean IoU (mIoU)
* Precision
* Recall
* F1 Score
* Per-class metrics

The main class of interest is:

```text
Clothes
```

because the primary objective of the project is accurate clothing segmentation.

---

# Inference on a Personal Image

The project also provides an inference script for segmenting clothing from a new personal image.

This workflow was tested in **Google Colab**.

## 1. Clone the repository

```python
!pip install uv --quiet

!git clone https://github.com/abdulrahman1238/clothes-segmentation-cvcy002

%cd /content/clothes-segmentation-cvcy002

!uv sync
```

## 2. Mount Google Drive

```python
from google.colab import drive

drive.mount('/content/drive')
```

## 3. Copy the trained model

Place the trained `best_model.pth` checkpoint in Google Drive and copy it into the repository:

```python
!cp "/content/drive/MyDrive/cvcy002/best_model.pth" \
    /content/clothes-segmentation-cvcy002/
```

## 4. Run prediction

Provide the path to the input image and the desired output path:

```python
!MPLBACKEND=Agg uv run python scripts/predict.py \
    --checkpoint /content/clothes-segmentation-cvcy002/best_model.pth \
    --image_path /content/241_190379.jpg \
    --output_path outputs/my_prediction2.png
```

The resulting segmentation image will be saved to:

```text
outputs/my_prediction2.png
```

---

# Reproducing the Results

The complete reproduction pipeline is:

```text
Clone repository
       ↓
Install dependencies
       ↓
Prepare / download LIP dataset
       ↓
Run LIP preprocessing
       ↓
Train DeepLabV3+
       ↓
Save checkpoints
       ↓
Evaluate best checkpoint
       ↓
Analyze segmentation metrics
       ↓
Run inference on new images
```

### Training command

```bash
uv run python scripts/train.py --config configs/config.yaml
```

### Evaluation command

```bash
uv run python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth
```

### Prediction command

```bash
uv run python scripts/predict.py \
    --checkpoint /path/to/best_model.pth \
    --image_path /path/to/image.jpg \
    --output_path outputs/prediction.png
```

For the most reproducible environment, use the dependency versions recorded in `uv.lock`.

---

# Pretrained Weights

The trained **best model checkpoint** is available here:

**[Download `best_model.pth`](https://drive.google.com/drive/u/0/folders/1Mfco8RY8W9NPgcdr2bkh08pEZcD_ZYsI)**

The checkpoint can be used directly with `scripts/predict.py` without retraining the model.

Example:

```bash
uv run python scripts/predict.py \
    --checkpoint /path/to/best_model.pth \
    --image_path /path/to/input.jpg \
    --output_path outputs/prediction.png
```

---

# Report

A detailed technical report describing the project methodology, including:

* Dataset selection and preprocessing
* Model architecture
* Loss function
* Evaluation metrics
* Performance analysis
* Limitations

is available here:

**[Project Report — PDF](https://drive.google.com/file/d/19I9iMKKoO0roFsAha5ZtRatK0ztglTzg/view?usp=sharing)**

---

# Notebooks

The repository provides three notebooks covering the main stages of the project: training, evaluation, and inference.

# 1. Training & Evaluation — Kaggle

Open cycv002-train-evaluate.ipynb

This notebook contains the main reproducible training workflow using Kaggle GPU.

It includes:

Installing uv
Cloning the repository
Installing project dependencies
Preparing and validating the LIP dataset
Running dataset preprocessing
Training DeepLabV3+
Saving model checkpoints
Resuming training when required
Evaluating the best checkpoint
Dataset requirement

Before running the notebook, attach the LIP Dataset on Kaggle to the Kaggle notebook as an Input/Data Source.

The dataset is not included in this repository.

# 2. Evaluation — Google Colab

Open cycv002_evaluate.ipynb

This notebook provides an additional evaluation workflow for the trained model.

It can be used to:

Load the trained checkpoint
Run evaluation
Calculate segmentation metrics
Analyze per-class performance
Inspect the model's predictions

The notebook provides an alternative evaluation pipeline to the command-line scripts/evaluate.py workflow.

# 3. Personal Image Inference — Google Colab

Open cycv002_predict.ipynb

This notebook demonstrates how to use the trained model on a personal image.

The workflow:

Clones the project repository.
Installs the required dependencies.
Mounts Google Drive.
Loads the trained best_model.pth checkpoint.
Provides a personal image as input.
Runs clothing segmentation.
Saves the predicted segmentation mask.

This notebook demonstrates the final intended use of the trained clothing segmentation model.
---


---

# References

1. Gong, K., Liang, X., Zhang, D., Shen, X., & Lin, L. **Look Into Person: Self-Supervised Structure-Sensitive Learning and a New Benchmark for Human Parsing.** CVPR 2017.
   https://openaccess.thecvf.com/content_cvpr_2017/html/Gong_Look_Into_Person_CVPR_2017_paper.html

2. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.** ECCV 2018.
   https://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html

3. uv — Python project and package management.
   https://docs.astral.sh/uv/

---

# Author

**Abdulrahman Hasan**

Computer Vision / Machine Learning Engineer

GitHub:
https://github.com/abdulrahman1238
