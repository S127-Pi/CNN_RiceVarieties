# CNN Rice Varieties Classification

This project aims to classify different varieties of rice grains using a Convolutional Neural Network (CNN). The model is trained to distinguish between multiple breeds, achieving high accuracy on the test dataset. The project includes scripts for data processing, model training, hyperparameter tuning, and interpretability using Grad-CAM.

## Project Structure

The repository is organized as follows:
CNN_RICEVARIETIES/
├── checkpoint/           # Directory for saving model checkpoints
├── data/                 # Directory for dataset storage
├── scripts/              # Shell scripts for project automation
│   └── test.sh           # Script to test the model
├── src/                  # Source code for the project
│   ├── analysis.ipynb    # Notebook for simple data analysis
│   ├── config.py         # Configuration file for project parameters
│   ├── dataset.py        # Script to handle data loading and processing
│   ├── earlystopping.py  # Early stopping callback for model training
│   ├── gradcam.ipynb     # Notebook for Grad-CAM visualizations
│   ├── hyperparameter.py # Hyperparameter tuning 
│   ├── main.py           # Main script for training and testing the model
│   ├── model.py          # Model setup
│   └── utils.py          # Utility functions
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
└── README.md             # Project documentation


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

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries can be installed via:
  ```pip install -r requirements.txt```