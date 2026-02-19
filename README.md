📊 Logistic Regression – Classification Project (Version 2)
📌 Overview

This project implements a Logistic Regression model to solve a supervised classification problem using Python and Scikit-learn. The notebook demonstrates the end-to-end machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), model training, prediction, and performance evaluation.

The primary goal is to build a classification model that predicts categorical outcomes accurately and evaluates performance using standard classification metrics.

🚀 Project Workflow
1️⃣ Import Required Libraries

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

2️⃣ Exploratory Data Analysis (EDA)

Dataset inspection

Checking for missing values

Visualizing feature distributions

Understanding relationships between variables

3️⃣ Data Preprocessing

Handling missing values

Encoding categorical variables

Feature selection

Splitting dataset into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

4️⃣ Model Building
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

5️⃣ Making Predictions
predictions = logmodel.predict(X_test)

6️⃣ Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

These metrics help evaluate how well the model performs on unseen data.

🛠 Technologies Used

Python

Jupyter Notebook

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn
