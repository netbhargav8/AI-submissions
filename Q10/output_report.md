# Output Report

## Project
**Hotel Booking Cancellation Prediction using Machine Learning**

---

## Data Preprocessing

- Loaded the Hotel Booking Demand dataset.
- Removed duplicate records.
- Handled missing values.
- Encoded categorical variables using Label Encoding.
- Split the dataset into training (80%) and testing (20%) sets.

---

## Model Used

**Random Forest Classifier**

The model was trained using the processed training dataset to predict booking cancellation status.

---

## Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- Classification Report

The confusion matrix was generated to analyze the prediction performance and identify correct and incorrect classifications.

---

## Feature Importance

The top 10 most influential features affecting booking cancellation were visualized using a horizontal bar chart.

Important features include:

- Lead Time
- Previous Cancellations
- Deposit Type
- ADR
- Special Requests

---

## Sample Prediction

The trained model was tested using a booking from the test dataset.

Example Output:

```
===== Testing on a Sample Booking =====

Actual Value    : CANCELLED

Predicted Value : CANCELLED

Result          :  Correct Prediction
```

*(The output may vary depending on the selected test sample.)*

---

## Saved Model

The trained model was successfully saved as:

```
hotel_booking_model.pkl
```

The label encoders were saved as:

```
label_encoders.pkl
```

---

## Conclusion

The Random Forest model successfully predicts hotel booking cancellations with good performance. The project demonstrates an end-to-end machine learning workflow, including data preprocessing, visualization, model training, evaluation, feature importance analysis, and prediction on unseen data.