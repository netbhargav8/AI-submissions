# Hotel Booking Cancellation Prediction

## 📌 Overview
This project uses a **Random Forest Classifier** to predict whether a hotel booking will be cancelled based on customer and booking details. It demonstrates a complete machine learning pipeline including data preprocessing, visualization, model training, evaluation, and model saving.

---

## 🚀 Features
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Label Encoding
- Random Forest Classification
- Model Evaluation
- Confusion Matrix
- Feature Importance Analysis
- Model Saving using Joblib

---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📂 Dataset
**Hotel Booking Demand Dataset**

Target Variable:
- **0** → Booking Not Cancelled
- **1** → Booking Cancelled

---

## 📊 Workflow

1. Load Dataset
2. Data Cleaning
3. Exploratory Data Analysis
4. Data Preprocessing
5. Train-Test Split
6. Train Random Forest Model
7. Evaluate Model
8. Feature Importance Analysis
9. Save Trained Model

---

## 📈 Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- Classification Report
- Confusion Matrix

---

## ▶️ Running the Project

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Run the notebook:

```bash
jupyter notebook hotel_booking.ipynb
```

---

## 💾 Saved Files

- `hotel_booking_model.pkl`
- `label_encoders.pkl`

---

## 📌 Business Insights

The analysis indicates that factors such as **lead time, previous cancellations, deposit type, ADR, and special requests** significantly influence booking cancellations. These insights can help hotels improve revenue management and reduce cancellation losses.

---
