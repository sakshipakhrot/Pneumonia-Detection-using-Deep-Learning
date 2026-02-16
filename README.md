# Pneumonia-Detection-using-Deep-Learning

# 🫁 Pneumonia Detection via Chest X-Rays: A Comparative Deep Learning Study

## 🏥 Project Motivation
Pneumonia is a critical respiratory infection that requires rapid and accurate diagnosis. While traditional CNNs are effective, medical datasets are often imbalanced or limited in size. This project conducts a comparative analysis between a **Custom CNN** and **MobileNetV2 Transfer Learning** to optimize diagnostic accuracy and reduce false-positive rates in clinical screening.

---

## 🔬 Experimental Methodology & Performance
We evaluated two distinct architectures on a dataset of chest X-ray images. The goal was to solve the "Recall Gap" observed in simpler architectures.

### 1. Custom CNN (Baseline)
A standard sequential architecture designed to learn spatial hierarchies from scratch.
* **Accuracy:** 72%
* **Key Issue:** High recall for Pneumonia (0.99) but very poor recall for Normal cases (0.28), indicating significant bias and "over-diagnosis."

### 2. MobileNetV2 (Transfer Learning)
Leveraged a pre-trained backbone on ImageNet, fine-tuned for binary medical classification.
* **Accuracy:** 88%
* **Key Improvement:** Achieved a balanced diagnostic profile with an **F1-score of 0.91** for Pneumonia and **0.82** for Normal cases.
* **Weighted Avg F1-Score:** 0.87 (vs. 0.67 for the baseline).



---

## 🛠️ Tech Stack & Libraries
* **Core:** Python
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV, PIL
* **Evaluation:** Scikit-learn (Classification Reports, Confusion Matrices)
* **Visualization:** Matplotlib, Seaborn

---

## ⚙️ Data Preprocessing & Augmentation
To mitigate overfitting and handle class imbalance, a robust image pipeline was implemented:
* **Normalization:** Pixel values scaled to `[0, 1]`.
* **Augmentation:** Real-time transformations including **Rotation, Zoom, Horizontal Flips, and Shear** using `ImageDataGenerator`.
* **Standardization:** Resized all X-ray images to `(224, 224)` to match the MobileNetV2 input requirements.
