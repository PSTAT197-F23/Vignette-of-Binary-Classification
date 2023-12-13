import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load the dataset
db = pd.read_csv("../../data/diabetes.csv")

# Remove rows where any column except 'Pregnancies' or 'Outcome' is 0
cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
db_filtered = db[(db[cols_to_check] != 0).all(axis=1)]

# Separate features and target variable
X = db_filtered.drop('Outcome', axis=1)
y = db_filtered['Outcome']

# Train Logistic Regression Classifier
lr_model = LogisticRegression(random_state=0, max_iter=1000)
cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')
print(f"Logistic Regression 5-Fold CV Accuracies:", cv_scores)
print("Mean Accuracy:", cv_scores.mean(),'\n')

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=0)
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"Random Forest 5-Fold CV Accuracies:", cv_scores)
print("Mean Accuracy:", cv_scores.mean(),'\n')

# Train Gradient Boosting Decision Trees (GBDT)
gbdt_model = GradientBoostingClassifier(random_state=0)
cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')
print(f"GBDT 5-Fold CV Accuracies:", cv_scores)
print("Mean Accuracy:", cv_scores.mean(),'\n')

# Train K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
cv_scores = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')
print(f"KNN 5-Fold CV Accuracies:", cv_scores)
print("Mean Accuracy:", cv_scores.mean(),'\n')

# Train XGBoost Classifier
xgb_model = XGBClassifier(random_state=0)
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
print(f"XGBoost 5-Fold CV Accuracies:", cv_scores)
print("Mean Accuracy:", cv_scores.mean(),'\n')
