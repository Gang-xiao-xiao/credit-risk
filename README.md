# 🛡️ Credit Card Fraud Detection Project

This project tackles the real-world problem of **credit card fraud detection** using both **unsupervised** and **supervised** machine learning approaches. The dataset used is the popular [Kaggle creditcardfraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 📦 Dataset Summary

- Total transactions: 284,807  
- Fraudulent transactions: 492 (~0.17%)  
- Features: 30 (PCA components + `Time` + `Amount`)  
- Extreme class imbalance (Legit vs Fraud ~ 577:1)  

---

## ⚙️ Key Techniques Used

### 1. **Exploratory Data Analysis**
- Visualized class distribution
- Checked for missing values
- Standardized `Time` and `Amount` features

### 2. **Model 1: Isolation Forest (Unsupervised)**
- Used for anomaly detection
- Trained only on input features (no labels)
- Evaluated with classification metrics after converting predictions

### 3. **Model 2: XGBoost Classifier (Supervised)**
- Applied `scale_pos_weight` to handle data imbalance
- Used `GridSearchCV` for hyperparameter tuning
- Optimized for **ROC-AUC**
- CPU training using `tree_method='hist'` for efficiency

---

## 🧪 Evaluation Metrics

- Classification Report (Precision / Recall / F1)
- ROC-AUC Score
- Confusion Matrix

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
