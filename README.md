# 🕵️‍♂️ Deepfake Detection: A Two-Phase Transfer Learning Approach

## 📌 Overview
With the rapid advancement of generative AI, detecting manipulated media has become a critical challenge in computer vision. This project implements a robust deep learning pipeline designed to accurately classify real versus deepfake videos/images. 

The core of the system relies on a **Two-Phase Transfer Learning** methodology, leveraging a pre-trained **ResNet50** backbone. By carefully implementing custom classification heads and executing a strategic fine-tuning process, the model achieves high sensitivity in identifying synthetic artifacts.

## ✨ Key Features & Methodology

### 1. Advanced Model Implementation
* **Base Architecture:** Utilized **ResNet50** for state-of-the-art spatial feature extraction.
* **Custom Classification Head:** Replaced the top layers of the pre-trained model with a custom fully connected network optimized for binary classification (Real vs. Fake), incorporating Dropout layers to prevent overfitting.
* **Data Preprocessing Pipeline:** Integrated facial extraction (cropping), normalization, and data augmentation techniques to ensure the model generalizes well to unseen faces and varying lighting conditions.

### 2. Two-Phase Fine-Tuning Strategy
To preserve the robust feature-extraction capabilities of the base model while adapting it to the specific domain of deepfake artifacts, the training was split into two distinct phases:
* **Phase 1 (Feature Extraction):** Froze the core ResNet50 layers and trained *only* the newly added custom dense layers. This allowed the classification head to learn the initial mapping without wrecking the pre-trained weights.
* **Phase 2 (Deep Fine-Tuning):** Unfroze the top convolutional blocks of the ResNet50 model. Using a significantly lower learning rate, the model was fine-tuned end-to-end. This step allowed the network to learn subtle, deepfake-specific pixel artifacts (like blending edges and warping).

### 3. Comprehensive Evaluation
* Evaluated model performance beyond standard accuracy using **AUROC (Area Under the Receiver Operating Characteristic curve)** and **F1-Score** to account for potential class imbalances and ensure high confidence in predictions.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch / PyTorch Lightning
* **Computer Vision:** OpenCV, PIL
* **Data Manipulation:** NumPy, Pandas
* **Machine Learning & Metrics:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to run this project in a GPU-accelerated environment (e.g., CUDA-enabled local machine or Google Colab).

```bash
# Clone the repository
git clone [https://github.com/yourusername/Deepfake-Detection.git](https://github.com/yourusername/Deepfake-Detection.git)
cd Deepfake-Detection
```
### Install required dependencies
```
pip install -r requirements.txt
```

R### unning the Notebook
The complete end-to-end implementation, including data loading, preprocessing, phase 1 & 2 training loops, and evaluation, is contained within the Jupyter Notebook.

```
jupyter notebook Deep_Fake_Detection.ipynb
```

### Results & Presentation
Detailed visualizations of the training curves (loss/accuracy), ROC curves, and prediction heatmaps can be generated via the notebook.
For a high-level overview of the project architecture, problem statement, and final metrics, please refer to the included `Presentation.pdf`.
