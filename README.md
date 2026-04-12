## AutoDetect — an AutoML framework for object detection that automatically trains ensemble YOLO models, optimizes hyperparameters with Optuna, and fuses predictions using Weighted Box Fusion (WBF) for state-of-the-art results.

## Example Notebook: [example Notebook](https://www.kaggle.com/code/antonoof/ad-autodetect-model-n?scriptVersionId=311074814)

## Features:

### 1. Ensemble Learning: Combines predictions from multiple YOLO architectures (yolov8, yolo11, yolo12) with different input resolutions.

### Optuna Hyperparameter Search: Automatically finds optimal conf_thresh, iou_thresh, and box filtering parameters for WBF.

### Weighted Box Fusion (WBF): Advanced box merging strategy (avg or max confidence aggregation) for robust inference.

### GPU-Accelerated: Built on PyTorch and Ultralytics for fast training and inference.

### Stratified Validation Sampling: Configurable validation subset for efficient hyperparameter tuning.

### Reproducibility: Full seed control and optional deterministic mode for consistent results.

## Installation:
```bash
pip install git+https://github.com/Antonoof/AutoDetect.git@main
```

## 🗂️ Dataset Structure:
```
dataset/
├── train/
│   ├── images/   # Training images (.jpg, .png, etc.)
│   └── labels/   # YOLO-format .txt labels (class_id x_center y_center w h)
├── val/
│   ├── images/   # Validation images
│   └── labels/   # Corresponding validation labels
└── test/
    └── images/   # Test images for inference (no labels required)
```

## Training: AutoDetect
```Python
from autodetect import AutoDetect

ad = AutoDetect(
    train="path/to/train",          # Path to train folder
    val="path/to/val",              # Path to val folder
    model='x',                      # Model size modifier: 'n', 's', 'm', 'l', 'x'
    epochs=30,                      # Number of training epochs
    model_config=(                  # Ensemble: (model_type, img_size)
        ('yolov8', 1024),
        ('yolo11', 960),
        ('yolo12', 768),
    ),
    device=None,                    # 'cpu', 0 or [0, 1] - two GPU
    warmup=False,                   # Enable RandomSearch to find best params YOLO
    inference_speed=-1,             # Target FPS for model selection (-1 = best accuracy)
    seed=42                         # Random seed for reproducibility
)

ad.fit()  # Start training and ensemble building
```

## Inference: ADPredict
```Python
from autodetect import ADPredict

predictor = ADPredict(
    image_paths=test_images,               # test dir (images in test dir)
    val_images_path=val_data,              # Validation images for Optuna tuning
    models_dir="result",                   # Directory with trained models default="result"
    output_dir="predictions",              # Where to save results
    device='auto',                         # 0, 'cpu', or 'auto'
    seed=42,
    deterministic=False,                   # Enable deterministic algorithms
    optuna_trials=30,                      # Number of Optuna hyperparameter trials
    val_samples=200,                       # Subset of val images for tuning
    wbf_conf_type="avg",                   # WBF confidence: 'avg' or 'max'
    conf_range=(0.005, 0.5),               # Search range for confidence threshold
    iou_range=(0.3, 0.7),                  # Search range for IoU threshold
    skip_box_range=(0.001, 0.05),          # Search range for box filtering
    model_weights=None                     # Optional: custom model weights example [0.9, 0.8, 1.3] for 3 models
)

# Run prediction
results = predictor.predict()
```

## Output Format:
#### Predictions are saved in:
- predictions/ — visualized images with bounding boxes
- predictions/results.csv — tabular results: [image_name, class_id, confidence, x1, y1, x2, y2]
- predictions/wbf_params.json — best WBF parameters found by Optuna

## Roadmap updates:
| Version | Planned Improvements |
|---------|---------------------|
| **v0.2.\*** | Enhanced hyperparameter stability, improved training convergence, better logging |
| **v0.3.\*** | GPU memory optimization, automatic model sizing based on image count/resolution, inference timing metrics |
| **v0.4.\*** | LR warmup scheduler, DINO-based validation split for stable training, time-aware hyperparameter search |
| **v0.5.\*** | Integration with [3LC](https://3lc.ai) or similar for active learning and bbox refinement |
| **v0.6.\*** | Advanced augmentation pipeline, Falcon-based synthetic image generation during training |
| **v0.7.\*** | SAHI integration for small-object detection, sliding-window inference support |
