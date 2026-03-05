# Nexa Aegis – AI-Assisted Mobile Forensics Platform for CSAM Detection

## Overview

**Nexa Aegis** is an AI-assisted mobile forensics platform designed to help investigators rapidly identify potential **Child Sexual Abuse Material (CSAM)** from digital evidence extracted from mobile devices.

Traditional forensic analysis can take significant time due to the large volume of media stored on modern smartphones. Nexa Aegis automates the first stage of analysis using computer vision models to detect NSFW content and estimate age, allowing investigators to prioritize high-risk material quickly.

The system is designed to assist digital forensic investigators by providing:

* Automated media screening
* Risk scoring for devices
* Metadata analysis
* Investigator dashboard for case review

This project was developed in collaboration with the **National Cyber Crime Investigation Agency (NCCIA)** as part of a Final Year Project in **Data Science at FAST-NUCES**.

---

## System Architecture

The Nexa Aegis pipeline processes evidence in multiple stages:

1. **Mobile Data Extraction**
2. **Media Processing and Filtering**
3. **NSFW Detection**
4. **Age Estimation**
5. **CSAM Risk Assessment**
6. **Investigator Dashboard**

Pipeline Flow:

```
Mobile Device
      │
      ▼
Data Extraction (ADB + ExifTool)
      │
      ▼
Media Preprocessing
      │
      ▼
NSFW Detection (YOLOv11 Segmentation)
      │
      ▼
Face Detection + Age Estimation
      │
      ▼
CSAM Risk Classification
      │
      ▼
Investigator Dashboard + Case Summary
```

---

## Key Features

### AI-Based NSFW Detection

A computer vision model detects explicit content using segmentation models trained on annotated datasets.

### Age Estimation

Detected faces are processed using a deep learning age estimation model trained on publicly available age-labeled datasets.

### CSAM Risk Detection Pipeline

If explicit content is detected and the estimated age is below 18, the media is flagged for potential CSAM review.

### Investigator Dashboard

The dashboard summarizes findings including:

* Risk score
* NSFW detections
* Possible CSAM indicators
* Media analysis statistics
* Metadata extracted from the device

---

## Dataset Creation

The dataset for NSFW detection was created using a **semi-supervised pipeline**.

### Initial Dataset

* 100k+ pornographic images collected from multiple sources

### Manual Annotation

A subset of images was manually labeled into four classes:

* Female genital
* Male genital
* Breast
* Anus

Dataset used for baseline training:

* 2,000 NSFW annotated images
* 2,000 normal images

### Pseudo-Labeling

A **Mask R-CNN baseline model** was trained on the manually annotated dataset.

Baseline model performance:

* Accuracy: **86%**

This model was used to generate pseudo-labels for the larger dataset.

### Human-in-the-Loop Verification

Pseudo-labeled images were manually reviewed and filtered to produce a balanced dataset of:

* **60,000 images**

Data augmentation techniques were applied to improve generalization.

---

## Model Training

### NSFW Detection Model

Final model:

* **YOLOv11 Segmentation**

Performance:

* Accuracy: **92%**

The model detects explicit body regions that are indicators of pornographic content.

---

### Age Estimation Model

A separate deep learning model was trained using publicly available datasets with labeled ages.

Training dataset characteristics:

* Wide age distribution
* Includes adolescent age groups
* Face-based age prediction

The age detection model estimates whether the subject is:

* Adult
* Minor
* Uncertain (requires human review)

---

## CSAM Detection Logic

The final decision pipeline works as follows:

```
Image Input
     │
     ▼
NSFW Detection
     │
     ├── No → Discard
     │
     ▼
Face Detection
     │
     ▼
Age Estimation
     │
     ├── Age ≥ 18 → Adult Content
     │
     ├── Age < 18 → Potential CSAM
     │
     └── Pubescent → Human Review
```

This approach ensures that **final decisions remain under human supervision**.

---

## Technologies Used

### Programming

* Python
* JavaScript

### Machine Learning

* PyTorch
* TensorFlow
* YOLO
* Mask R-CNN
* OpenCV

### Data Processing

* NumPy
* Pandas

### Mobile Forensics

* ADB
* ExifTool

### Dashboard

* Flask
* Electron

---

## Repository Structure

```
nexa-aegis-csam-detection
│
├── age_detection
│   ├── age_detection_pipeline.ipynb
│   ├── facedet-efficient.ipynb
│   └── face_age_test.py
│
├── nsfw_detection
│   ├── mask_rcnn_training.ipynb
│   └── verify_class_mappings.py
│
├── yolo_training
│   ├── train_kaggle.ipynb
│   ├── utils.py
│   └── TRAINING_CONFIG.md
│
├── dashboard
│
├── docs
│   ├── dataset_analysis.png
│   ├── training_curves.png
│   └── architecture.png
│
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/marwhyam/nexa-aegis-csam-detection.git
cd nexa-aegis-csam-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Important Note

Model weights are not included in the repository due to size limitations.

Pretrained weights can be downloaded separately.

---

## Ethical Considerations

This project was developed strictly for **digital forensic research and law enforcement assistance**.

Key ethical safeguards include:

* Human-in-the-loop verification
* Restricted dataset access
* Privacy-aware design
* Intended use only for legal investigation

