# Vietnamese Sign Language Detector

This repository contains the source code for a 100-class Vietnamese Sign Language classification project. It was developed as part of the **Vietnam Collegiate Olympiad in Artificial Intelligence 2025 (Selection Round). 

The core architecture utilizes **TimeSformer** (Time-Space Transformer) combined with various data augmentation techniques to accurately recognize and classify sign language vocabulary from video sequences.

## 📊 Results

The model has been evaluated on the AI Challenge platform and achieved the following metrics:
* **Validation F1-Score:** 89.31%
* **Private Test Accuracy:** 68.1%

## 📁 Repository Structure

```text
sign-language-detector/
├── scripts/                    # Executable scripts for running tasks
│   ├── __init__.py
│   ├── demo.py                 # Interactive demo script (Webcam/Video)
│   ├── evaluate.py             # Script for model evaluation
│   └── train.py                # Script to start training
├── src/                        # Core source code modules
│   ├── __init__.py
│   ├── config.py               # Configuration parameters
│   ├── dataset.py              # Data loading and preprocessing pipelines
│   ├── inference.py            # Inference logic
│   ├── model.py                # TimeSformer model architecture definitions
│   ├── train.py                # Training loop functions
│   └── utils.py                # Helper functions and utilities
├── .gitignore
├── best_model_timesformer.pth  # Pre-trained model weights
├── label_mapping.pkl           # Mapping of 100 classes to their Vietnamese labels
├── LICENSE                     # MIT License
├── README.md                   
└── requirements.txt            # Python dependencies
```

## ⚙️ Prerequisites & Installation

* **Python Version:** Python 3.10 or higher is required.
* It is highly recommended to use [Conda](https://docs.conda.io/en/latest/miniconda.html) to create an isolated virtual environment.

**Step 1: Create and activate a Conda environment**
```bash
conda create -n sign_lang python=3.10
conda activate sign_lang
```

**Step 2: Install required dependencies**
```bash
pip install -r requirements.txt
```

## 🗂️ Dataset & Baseline

The dataset and baseline code used in this project are the intellectual property of the **Posts and Telecommunications Institute of Technology (PTIT)**. 

To run the training and evaluation scripts, you must first download the dataset from the AI Challenge platform. It can be accessed through the following link:
* [AI Challenge PTIT - Competition 77](https://aichallenge.ptit.edu.vn/competitions/77/?fbclid=IwY2xjawQu29NleHRuA2FlbQIxMABicmlkETFtRTRWODJiTEdHSUlmQ3Vwc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHnKzDG_oRblcbm0lWlFRgEviwCRQtSuZWSTd3BKF89s4ADqGzCnWMtCOV7DJ_aem_KtSa0oOkAQH0eD1uepoLWQ#/phases-tab)

## 🚀 Usage

### 1. Running the Demo

The project includes an interactive demo script that supports two modes: **Webcam real-time detection** and **Video file prediction**. 

To launch the demo, simply run the script. An interactive menu will appear in the console allowing you to select your preferred mode without needing to pass terminal arguments:

```bash
python scripts/demo.py
```
* **Mode 1 (Video Path):** You will be prompted to enter the absolute or relative path to a `.mp4` video file.
* **Mode 2 (Webcam):** The system will open your default camera and perform real-time sign language recognition.

### 2. Training and Evaluation

After downloading the dataset and placing it in the correct directory (configured in `src/config.py`), you can train or evaluate the model using:

```bash
# To evaluate the model
python scripts/evaluate.py

# To train the model from scratch
python scripts/train.py
```

## 📄 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details. 
*(Note: This license applies to the code developed in this repository. The dataset remains the property of PTIT).*