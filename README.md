# 🔬 Breast Cancer Classification

**DecodeLabs Project 2** | **Batch 2026** | **AI Internship**

---

## 📌 Overview

This project builds a **Machine Learning model** to classify breast tumors as **Benign** (non-cancerous) or **Malignant** (cancerous) using 30 diagnostic features.

Unlike the standard Iris classification problem, this project utilizes the **Breast Cancer Wisconsin Dataset** — a real-world medical dataset containing **569 samples** and **30 features**, making it more practical and impactful.

### ✨ Key Features

* ✅ Comparison of **5 Machine Learning algorithms**
* ✅ Hyperparameter tuning using **GridSearchCV**
* ✅ Interactive **Streamlit web application**
* ✅ Model evaluation using multiple metrics
* ✅ Feature importance visualization
* ✅ Achieved **98.25% accuracy** with Random Forest

---

## 📊 Dataset Information

| Property       | Details                             |
| -------------- | ----------------------------------- |
| Source         | UCI Breast Cancer Wisconsin Dataset |
| Samples        | 569                                 |
| Features       | 30                                  |
| Classes        | Benign (357), Malignant (212)       |
| Missing Values | None                                |

---

## 🤖 Model Performance

| Model                        | Accuracy      |
| ---------------------------- | ------------- |
| K-Nearest Neighbors (KNN)    | 96.49%        |
| Support Vector Machine (SVM) | 97.37%        |
| Decision Tree                | 95.61%        |
| **Random Forest**            | **98.25%** 🏆 |
| Logistic Regression          | 97.37%        |

🏆 **Best Model:** Random Forest with **98.25% Accuracy**

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-classifier.git
cd breast-cancer-classifier
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python train_model.py
```

### 4️⃣ Launch the Web Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
breast-cancer-classifier/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
└── assets/
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 🛠️ Technologies Used

* Python 3.7+
* Scikit-learn
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Joblib

---

## 🎯 Learning Outcomes

Through this project, I gained hands-on experience in:

* Building an end-to-end supervised learning pipeline
* Hyperparameter tuning using GridSearchCV
* Cross-validation for robust evaluation
* Data preprocessing and feature scaling
* Model deployment using Streamlit
* Version control using Git and GitHub

---

## 👨‍💻 Author

**Payal Priya**


---


