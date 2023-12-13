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
    print(f"\n{name} Metrics:")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")   
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
    plt.savefig(f"../../img/{filename}-default")


# Train Logistic Regression Classifier
rf_model = LogisticRegression(max_iter=1000)
rf_model.fit(X_train, y_train)
compute_metrics_and_plot(rf_model, X_test, y_test, "Logistic Regression","LogR")

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)
compute_metrics_and_plot(rf_model, X_test, y_test, "Random Forest","RanF")

# Train Gradient Boosting Decision Trees (GBDT)
gbdt_model = GradientBoostingClassifier(random_state=0)
gbdt_model.fit(X_train, y_train)
compute_metrics_and_plot(gbdt_model, X_test, y_test, "Gradient Boosted Trees","GBDT")

# Train K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
compute_metrics_and_plot(knn_model, X_test, y_test, "K-Nearest Neighbors","KNN")


# Train XGBoost Classifier
xgb_model = XGBClassifier(random_state=0,learning_rate=0.3)
xgb_model.fit(X_train, y_train)
compute_metrics_and_plot(xgb_model, X_test, y_test, "XGBoost","XGB")
