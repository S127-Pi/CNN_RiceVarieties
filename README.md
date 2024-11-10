# CNN Rice Varieties Classification

This project aims to classify different varieties of rice grains using a Convolutional Neural Network (CNN). The model is trained to distinguish between multiple breeds, achieving high accuracy on the test dataset. The project includes scripts for data processing, model training, hyperparameter tuning, and interpretability using Grad-CAM.

### Folder and File Descriptions

- **checkpoint/**: Stores model checkpoints, which are saved states of the model during training and testing.
- **data/**: This folder is where the dataset is stored and organized for training and testing purposes.
- **scripts/**: Contains scripts for automated testing.
- **src/**: Core directory containing the source code and essential files for the project.
  - **config.py**: Stores configuration settings such as hyperparameters and data paths.
  - **dataset.py**: Manages dataset loading, transformations, and preprocessing.
  - **earlystopping.py**: Implements early stopping to prevent overfitting during model training.
  - **gradcam.ipynb**: Notebook for visualizing Grad-CAM to interpret the model's predictions.
  - **hyperparameter.py**: Script for hyperparameter tuning to improve model performance.
  - **main.py**: Main entry point for training and testing the model.
  - **model.py**: Defines the model.
  - **utils.py**: Contains various utility functions used throughout the project.
- **requirements.txt**: Contains dependencies for the project,

## Getting Started

### Prerequisites

- Python 3.8 or newer
- Required libraries can be installed via:
  ```pip install -r requirements.txt```
