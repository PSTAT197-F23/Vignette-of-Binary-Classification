from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset
db = pd.read_csv("../../data/diabetes.csv")

# Remove rows where any column except 'Pregnancies' or 'Outcome' is 0
cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
db_filtered = db[(db[cols_to_check] != 0).all(axis=1)]

# Separate features and target variable
X = db_filtered.drop('Outcome', axis=1)
y = db_filtered['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to compute metrics and plot ROC curve
def compute_metrics_and_plot(model, X_test, y_test, name, filename):
    print(f"\n{name} with Best Params Metrics:")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")   
    # Compute ROC curve and ROC area for each class
    y_scores = model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC of {name}')
    plt.legend(loc="lower right")
    plt.savefig(f"../../img/{filename}-tuned")

# GridSearchCV for Logistic Regression
logistic_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                   'penalty': ['l1', 'l2']}
logistic_grid = GridSearchCV(LogisticRegression(max_iter=1000,solver='liblinear'), param_grid=logistic_params, cv=5)
logistic_grid.fit(X_train, y_train)
best_logistic = logistic_grid.best_estimator_
print("Best parameters for Logistic Regression:", logistic_grid.best_params_)
compute_metrics_and_plot(best_logistic, X_test, y_test, "Tuned Logistic Regression", "LogR")

# GridSearchCV for Random Forest
rf_params = {
            'max_depth': [3, 4, 5, 6]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid=rf_params, cv=5)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("Best parameters for Random Forest:", rf_grid.best_params_)
compute_metrics_and_plot(best_rf, X_test, y_test, "Tuned Random Forest","RanF")

# GridSearchCV for Gradient Boosting Decision Trees (GBDT)
gbdt_params = {
               'learning_rate': [0.01, 0.1, 0.2, 0.3],
               'max_depth': [3, 4, 5, 6]}
gbdt_grid = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid=gbdt_params, cv=5)
gbdt_grid.fit(X_train, y_train)
best_gbdt = gbdt_grid.best_estimator_
print("Best parameters for GBDT:", gbdt_grid.best_params_)
compute_metrics_and_plot(best_gbdt, X_test, y_test, "Tuned Gradient Boost Trees","GBDT")

# GridSearchCV for K-Nearest Neighbors (KNN)
knn_params = {'n_neighbors': [3, 5, 7, 9],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, cv=5)
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
print("Best parameters for KNN:", knn_grid.best_params_)
compute_metrics_and_plot(best_knn, X_test, y_test, "Tuned K Nearest Neighbors", "KNN")

# GridSearchCV for XGBoost Classifier
xgb_params = {
              'max_depth': [3, 4, 5, 6]}
xgb_grid = GridSearchCV(XGBClassifier(random_state=0, learning_rate=0.3), param_grid=xgb_params, cv=5)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print("Best parameters for XGBoost:", xgb_grid.best_params_)
compute_metrics_and_plot(best_xgb, X_test, y_test, "Tuned XGBoost","XGB")

