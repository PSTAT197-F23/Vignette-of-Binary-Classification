# Functions


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
