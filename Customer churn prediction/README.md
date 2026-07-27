# Customer Churn Prediction using Machine Learning

## Project Overview

This project predicts whether a bank customer is likely to leave (churn)
based on demographic and banking information. It uses supervised machine
learning classification techniques to help improve customer retention.

## Dataset

**File:** `Churn_Modelling.csv`

### Features

-   RowNumber
-   CustomerId
-   Surname
-   CreditScore
-   Geography
-   Gender
-   Age
-   Tenure
-   Balance
-   NumOfProducts
-   HasCrCard
-   IsActiveMember
-   EstimatedSalary
-   Exited (Target)

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Streamlit

## Workflow

1.  Import dataset
2.  Data preprocessing
3.  Encode categorical variables
4.  Feature scaling
5.  Train-test split
6.  Train classification model
7.  Evaluate performance
8.  Predict customer churn

## Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   Confusion Matrix
-   ROC-AUC

## Project Structure

``` text
Customer-Churn-Prediction/
├── Churn_Modelling.csv
├── Customer_Churn_Prediction.ipynb
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

## Installation

``` bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

-   Hyperparameter tuning
-   SHAP explainability
-   Cloud deployment
-   Interactive dashboard

## Author

**Aditi Gawade**\
AI & DS Engineering Student
