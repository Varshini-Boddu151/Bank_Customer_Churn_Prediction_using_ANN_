**Bank Customer Churn Prediction using ANN**

A deep learning project that predicts whether a bank customer is likely **stay with the bank or churn**, built with an Artificial Neural Network (ANN) in TensorFlow/Keras, and deployed as an interactive Streamlit web app.

***📌 Overview***

Customer churn is costly for banks — retaining an existing customer is far cheaper than acquiring a new one. This project builds a binary classifier that flags customers at risk of leaving, so the bank can intervene early.

Dataset: Bank Customer Churn dataset — 10,000 customers, 14 features (credit score, geography, gender, age, tenure, balance, products held, credit card status, activity status, salary), with a binary target Exited (1 = churned, 0 = stayed).

***🚀 Live Demo***

Run locally with Streamlit — see Setup below. The app takes customer details through a form and returns a churn prediction with probability.

***🧠Project Pipeline***

1. **Data Cleaning** — dropped non-predictive columns (RowNumber, CustomerId, Surname), standardized column names

2. **Exploratory Data Analysis** — class distribution, feature distributions, correlation heatmap

3. **Encoding** — one-hot encoded categorical features (geography, gender) via pd.get_dummies

4. **Train/Val/Test Split** — 70/15/15 stratified split to preserve the churn ratio across all sets

5. **Feature Scaling** — StandardScaler fit on train only, applied to val/test (no data leakage)

6. **Class Imbalance Handling** — computed class weights (~80/20 imbalance) so the model doesn't just predict the majority class

7. **Model Architecture** — regularized ANN with L2 regularization, Dropout, and BatchNormalization to prevent overfitting

8. **Training** — Adam optimizer, binary cross-entropy loss, EarlyStopping + ReduceLROnPlateau callbacks

9. **Threshold Tuning** — found the probability cutoff that maximizes F1 on the validation set, instead of using the default 0.5

10. **Evaluation** — final, one-time evaluation on the held-out test set

11. **Deployment** — saved model/scaler/metadata as artifacts, wrapped in a Streamlit app for live predictions
    



***📊 Results***

Metric	Score
Test Accuracy	84.2%
Test ROC-AUC	0.854
Precision (churn class)	0.61
Recall (churn class)	0.60
F1 (churn class)	0.61
Decision threshold	0.63 (tuned, not default 0.5)




***🛠️ Tech Stack***

**Python, Pandas, NumPy** — data handling

**Scikit-learn** — preprocessing, splitting, metrics, class weights

**TensorFlow / Keras** — ANN model

**Matplotlib / Seaborn** — visualization

**Streamlit** — web app deployment





**👩‍💻 Project Highlights**

This project demonstrates an end-to-end Artificial Neural Network-based customer churn prediction system, including:

EDA → Preprocessing → Encoding → Scaling → Class Weighting → ANN → Regularization → Early Stopping → Threshold Tuning → Evaluation → Model Saving





**⭐ Author**

B.Varshini





**Skills Demonstrated**

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras, Artificial Neural Networks, Deep Learning, EDA, Classification
F1 (churn class)	0.61
Decision threshold	0.63 (tuned, not default 0.5)
