# YOLOv8 Column Detection Model

## Overview

This repository contains the training pipeline for a YOLOv8-based object detection model specialized in detecting columns in images. The project implements an automated training workflow using the Ultralytics YOLOv8 framework to train a medium-sized model on a curated dataset of 850 annotated images.

## Context

This repository is part of a broader computer vision project focused on automated structural element detection. It specifically handles the model training and optimization phase, preparing a deep learning model that can identify and localize columns in various contexts. The trained model can be deployed in applications requiring structural analysis, architectural documentation, or automated building inspection systems.

## Technologies

- **YOLOv8 (Ultralytics)**: Object detection framework and model architecture
- **Python**: Core programming language for training pipeline
- **Roboflow**: Dataset management and preprocessing platform
- **CUDA/GPU**: Hardware acceleration for training (device 0 specified in configuration)

## Architecture

The project implements a straightforward, production-focused training architecture:

- **Training Script** (`treino_yolo.py`): Entry point that initializes the YOLOv8 model and orchestrates the training process
- **Dataset Configuration** (`data.yaml`): YAML configuration specifying dataset paths, class definitions, and dataset metadata
- **Dataset**: 850 images organized into train/validation/test splits, sourced from Roboflow

### Training Pipeline

The training workflow:
1. Loads a pre-trained YOLOv8 medium model (`yolov8m.pt`)
2. Reads dataset configuration from `data.yaml`
3. Trains for 250 epochs with early stopping (patience: 50 epochs)
4. Uses a batch size of 24 with input resolution of 512×512 pixels
5. Saves checkpoints and training artifacts to the `runs/` directory

## Features

- Automated YOLOv8 model training with configurable hyperparameters
- GPU-accelerated training with CUDA support
- Early stopping implementation with 50-epoch patience threshold
- Cached image loading for improved training performance
- Single-class object detection (column detection)
- Integration with Roboflow dataset management

## Repository Structure

```
train-yolo-model/
├── treino_yolo.py              # Main training script
├── Column.v2i.yolov8/          # Dataset configuration and metadata
│   ├── data.yaml               # YOLOv8 dataset configuration
│   ├── README.dataset.txt       # Dataset documentation
│   ├── README.roboflow.txt      # Roboflow dataset information
│   ├── train/                  # Training images (gitignored)
│   └── valid/                  # Validation images (gitignored)
├── runs/                        # Training outputs (gitignored)
└── .gitignore                  # Version control exclusions
```

**Key Directories:**
- `Column.v2i.yolov8/`: Contains dataset configuration and references to train/validation/test splits
- `runs/`: Auto-generated directory containing trained models, checkpoints, and training metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)
- pip package manager

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd train-yolo-model
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics
   ```

4. **Prepare the dataset:**
   - Download the dataset from Roboflow or ensure it's available at `Column.v2i.yolov8/train` and `Column.v2i.yolov8/valid`
   - The configuration expects the following structure:
     ```
     Column.v2i.yolov8/
     ├── train/images/
     ├── valid/images/
     └── test/images/
     ```

## Usage

### Training the Model

Execute the training script:

```bash
python treino_yolo.py
```

**Training Configuration:**
- **Model**: YOLOv8 Medium (`yolov8m`)
- **Epochs**: 250
- **Batch Size**: 24
- **Input Size**: 512×512 pixels
- **Device**: GPU (device 0)
- **Early Stopping**: 50 epochs without improvement
- **Image Cache**: Enabled for faster training

**Expected Output:**

Training progress and metrics will be saved to the `runs/` directory, containing:
- Trained model weights
- Training curves and metrics
- Validation results
- Tensorboard logs (if enabled)

### Monitoring Training

After training completes, results are stored in `runs/detect/train<N>/`. Review:
- `results.csv`: Training and validation metrics
- `confusion_matrix.png`: Model confusion matrix
- `best.pt`: Best model checkpoint (recommended for deployment)

## My Contributions

Based on repository evidence:

- **Model Selection and Configuration**: Implemented YOLOv8 medium model training pipeline with optimized hyperparameters (512×512 input resolution, batch size 24)
- **Training Infrastructure**: Developed `treino_yolo.py` to automate the model training workflow with GPU acceleration
- **Dataset Integration**: Configured dataset handling through `data.yaml` with support for train/validation/test splits
- **Training Optimization**: Applied caching and multi-worker data loading (2 workers) for efficient training throughput
- **Early Stopping Strategy**: Implemented patient-based early stopping (50 epochs) to prevent overfitting

## Future Improvements

- **Model Ensemble**: Experiment with ensemble methods combining YOLOv8 variants (small, medium, large) for improved robustness
- **Data Augmentation**: Implement advanced augmentation techniques (mosaic, mixup, rotation) to improve model generalization
- **Hyperparameter Optimization**: Conduct automated hyperparameter tuning using Bayesian optimization or grid search
- **Inference Optimization**: Implement model quantization and pruning for deployment on edge devices
- **Real-time Inference**: Develop API endpoints for real-time column detection with confidence scoring
- **Cross-validation**: Implement k-fold cross-validation to ensure model robustness
- **Transfer Learning**: Experiment with different backbone architectures and pre-trained weights

## Team

This project was developed by a team of Computer Science students as part of an academic computer vision initiative. The collaborative effort demonstrates competency in deep learning model training, dataset management, and practical implementation of state-of-the-art object detection frameworks.

---

**Dataset License**: CC BY 4.0  
**Dataset Source**: [Roboflow Universe - Column Detection](https://universe.roboflow.com/pi-eeksi/column-2i5mg-ypuqv/dataset/2)  
**Model Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)