# Lung Disease Classification using CNNs & Transfer Learning

## Project Overview
This project focuses on the multi-label classification of chest X-ray images to detect common thoracic diseases. Using the **NIH Chest X-ray Dataset**, the model is designed to identify four specific conditions: **Atelectasis, Effusion, Infiltration, and No Finding**.

The project evolves from a custom baseline CNN to a highly optimized solution using **Hyperparameter Tuning (Keras Tuner)**, **Transfer Learning (VGG16)**, and **Weighted Ensemble methods**, achieving a Micro Average AUC-ROC of **0.776**.

## Key Features
* **Data Analysis:** Visualized label co-occurrences using NetworkX to understand disease correlations.
* **Data Augmentation:** Implemented `RandomRotation`, `RandomContrast`, and `RandomZoom` to improve model generalization.
* **Hyperparameter Tuning:** Used **Keras Tuner (RandomSearch)** to optimize the number of convolutional layers, dense units, and learning rates.
* **Transfer Learning:** Fine-tuned a **VGG16** model pre-trained on ImageNet.
* **Ensemble Modeling:** Implemented a weighted average ensemble of the top 3 models to maximize performance.

## Dataset
* **Source:** NIH Chest X-ray Dataset.
* **Preprocessing:**
    * Images resized to `256x256`.
    * Multi-Label Binarizer used for One-Hot Encoding of target labels.
    * Filtered to focus on high-frequency classes: `No Finding`, `Infiltration`, `Effusion`, `Atelectasis`.

## Methodology & Models

### 1. Custom CNN (Baseline vs. Regularized)
Initial experiments involved a custom Sequential CNN. Overfitting was addressed using:
* **Dropout layers** (0.25 - 0.5 rates).
* **L2 Regularization**.
* **Batch Normalization**.

### 2. Keras Tuner Optimization
Automated the architecture search to find the optimal depth and width.
* *Search Space:* 2-4 Convolutional blocks, 32-128 filters, varying dropout rates.
* *Result:* The tuner identified a 3-layer architecture with 128 dense units and a learning rate of `0.001` as optimal.

### 3. Transfer Learning (VGG16)
Leveraged the feature extraction capabilities of VGG16.
* **Input:** Adjusted for grayscale (1-channel) X-rays.
* **Head:** Custom fully connected layers added on top of the frozen VGG base.
* **Performance:** This model outperformed the custom CNNs significantly in AUC scores.

### 4. Ensemble Approach
To squeeze out the best performance, predictions were combined using a weighted average:
```python
# Weighted Average Ensemble
y_pred_ensemble = (0.70 * y_pred_vgg16) + (0.25 * y_pred_keras_tuned) + (0.05 * y_pred_best)
```

## Results
The Ensemble model achieved the best results:

| Metric | Score |
| :--- | :--- |
| **Micro AUC-ROC** | **0.7759** |
| **Macro AUC-ROC** | **0.7605** |
| Infiltration AUC | 0.8675 |
| Atelectasis AUC | 0.7716 |
| No Finding AUC | 0.7559 |
| Effusion AUC | 0.6468 |


### Technologies Used

- Python

- TensorFlow / Keras

- Pandas & NumPy

- Matplotlib & Seaborn

- NetworkX (Graph visualization)


### How to Run

1. Clone the repository:
```
git clone [https://github.com/alecjanderson/lung-disease-cnn-portfolio.git](https://github.com/alecjanderson/lung-disease-cnn-portfolio.git)
```

2. Install dependencies
```
pip install tensorflow pandas numpy matplotlib networkx keras-tuner
```

Navigate to the notebooks directory and launch Jupyter
```
cd notebooks
jupyter notebook
```
