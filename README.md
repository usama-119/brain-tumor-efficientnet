Brain Tumor Classification using EfficientNetB0 & EfficientNetB1
Overview
This project implements a deep learning model using EfficientNetB0 and EfficientNetB1 to classify brain tumors from MRI images. The model is trained to distinguish between four categories of brain tumors:

Glioma
Meningioma
Pituitary
No Tumor
Dataset
The dataset used for training and evaluation is the Brain Tumor MRI Dataset, structured as follows:

Brain Tumor MRI Dataset/
  ├── Training/
  │   ├── Glioma/
  │   ├── Meningioma/
  │   ├── Pituitary/
  │   ├── No Tumor/
  ├── Testing/
      ├── Glioma/
      ├── Meningioma/
      ├── Pituitary/
      ├── No Tumor/
📂 Dataset & Preprocessing
Dataset: Brain MRI scans categorized into four tumor types.

Preprocessing steps:

Duplicate images removed for better dataset integrity.

Image resizing and normalization.

Data augmentation to improve generalization.

Splitting into training, validation, and test sets.

🔍 Model Architecture
We utilized EfficientNetB0 and EfficientNetB1, which are lightweight and high-performing convolutional neural networks (CNNs). The models were fine-tuned on the dataset to achieve high accuracy while maintaining efficiency.

Training Process
Data was preprocessed and normalized.
Images were augmented to improve generalization.
The dataset was split into training and validation sets.
The models were trained using TensorFlow and Keras.
📊 Performance & Evaluation
EfficientNetB0
Training & Validation Accuracy!
EfficientNetB0 (Train   Val acc )

Confusion Matrix
EfficientNetB0 Confusion Matrix

Random Predictions
Random Prediction on images (EfficientNetB0)

EfficientNetB1
EfficientNetB1
Training & Validation Accuracy
(EfficientNetB1 (Train   Val acc ))

Confusion Matrix
EfficientNetB1 Confusion Matrix

Random Predictions
Random Prediction on images (EfficientNetB1)

🚀 Results
EfficientNetB0 achieved a validation accuracy of ~80%.
EfficientNetB1 performed significantly better, reaching ~98% accuracy.
The confusion matrices show strong performance in identifying tumor types correctly.
Requirements
To run the model, install the required dependencies:

pip install tensorflow keras numpy matplotlib seaborn
How to Use
Clone this repository:
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
Run the training script:
python train.py
Test the model on new images:
python predict.py --image path/to/image.png
Conclusion
EfficientNet models demonstrated strong classification capabilities for brain tumor MRI scans. EfficientNetB1 outperformed EfficientNetB0, making it the preferred choice for real-world applications.

Future Improvements
Experimenting with EfficientNetB2-B7 for better accuracy.
Integrating Grad-CAM to visualize model decisions.
Deploying the model as a web application for easy access.
Author: Usama Riaz
