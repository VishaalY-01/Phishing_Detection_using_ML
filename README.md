# PHISHING_DETECTION_USING_ML

Machine Learning-based URL Phishing Detection Tool

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://github.com/VishaalY-01/Phishing_Detection_using_ML)

---

## Table of Contents

- Overview
- Features
- Installation
- Usage
- Project Structure
- Models and Algorithms
- Contributing
- License
- Contact

---

## Overview

This project implements a robust URL phishing detection system using machine learning techniques. It processes URL data to classify links as phishing or legitimate
based on multiple classifiers including Naive Bayes, Support Vector Machines (SVM), and Deep Learning models built with Keras/TensorFlow.

## Features

- Text preprocessing and feature extraction from URL datasets.
- Multiple classifiers to improve detection accuracy:
  - Naive Bayes
  - Support Vector Machine
  - Deep Learning Neural Network
- Model training, testing, and performance evaluation.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/VishaalY-01/Phishing_Detection_using_ML.git
```

2. Navigate to the project directory:

```bash
cd Phishing_Detection_using_ML
```

3. Install required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run `project.py` to execute training and evaluation:

```bash
python project.py
```

Modify the script if you want to test with custom URLs or datasets.

## Project Structure

```
Phishing_Detection_using_ML/
├── phishing_dataset.csv    # Dataset of URLs with labels
├── phishing_model.keras    # Saved Keras deep learning model
├── project.py              # Main script: training & testing
└── README.md               # Project documentation
```

## Models and Algorithms

- **Naive Bayes classifier:** Multinomial Naive Bayes on vectorized URL text.
- **Support Vector Machine:** Linear kernel SVM for binary classification.
- **Deep Learning model:** Fully connected neural network using Keras with 3 dense layers.

## Contributing

Contributions are welcome! Please open a pull request.

## License

This project is licensed under the MIT License.

## Contact

Feel free to open issues or contact me via GitHub.

---

Thank you for using PHISHING_DETECTION_USING_ML to help enhance online security!
