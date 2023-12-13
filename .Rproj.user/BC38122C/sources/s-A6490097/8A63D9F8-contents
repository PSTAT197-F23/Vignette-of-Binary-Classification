Vignette on implementing Binary Classification using patients' medical data in order to classify/predict if a patient has diabetes; Group 12 created as a class project for PSTAT197A in Fall 2023.

Contributors:
Group 12: Erica Chen, Chris Straw, and Claire Lee

Vignette abstract:
For our vignette class project, we decided to choose the topic Nonlinear Binary Classification, and specifically cover the binary classification algorithms: Logisitc Regression, Random Forest, Boosted Tree Model, K-Nearest Neighbors, Gradient Boosting Decision Trees (GBDT), and XGBoost. Binary Classification is a method in machine learning where we can classify an observation to be under one of the two class labels based on predictors of the given data. The classes are generally labeled as 0 and 1, but can also use the labels: True and False, or positive and negative. 

The example data we used to carry out these different binary classification algorithms sourced from the National Institute of Diabetes and Digestive Kidney Disease. This dataset is patients' data of females at least twenty one years old and of Pima Indian heritage. There are eight independent variables: Pregnancies, Glucose Concentration, Blood Pressure, Skin Thickness, Insulin Levels, BMI (Body Mass Index), Diabetes Pedigree Function, and Age. The response variable is Outcome where 1 means Yes diabetes and 0 is No diabetes. Our primary objective is to develop a binary classification model that accurately predicts whether a female of Pima Indian heritage who is at least 21 years old has diabetes or not. 

We then compare the different binary classification algorithms and choose the final model based on performance metrics such as accuracy, precision, recall, and F1 score. For the Logistic Regression, the accuracy was 0.7722, the precision was 0.6957, the recall was 0.5926, and the F1 score was 0.6400. For the Random Forest, the accuracy was 0.7848, the precision was 0.7083, the recall was 0.6296, and the F1-score was 0.6667. For the K-Nearest Neighbors, the accuracy was 0.7089, the precision was 0.5769, the recall was 0.5556, and the F1-score was 0.5660. For the XGBoost, the accuracy was 0.7468, the precision was 0.6400, the recall was 0.5926, and the F1-score was 0.6154. The GBDT accuracy measures were the same as the XGBoost.

To improve the performance of the models and increase the accuracy measures, we then use 5-fold Cross Validation. 
For the Logistic Regression 5-Fold CV, the accuracies are [0.83544304 0.70886076 0.74358974 0.82051282 0.82051282] with a Mean Accuracy of 0.7857838364167478. For Random Forest 5-Fold CV, the accuracies are [0.7721519  0.6835443  0.79487179 0.80769231 0.82051282] with a Mean Accuracy of 0.7757546251217138. For GBDT 5-Fold CV, the Accuracies are [0.83544304 0.70886076 0.4358974 0.82051282 0.82051282] with a Mean Accuracy of 0.7857838364167478. The KNN 5-Fold CV Accuracies are [0.73417722 0.67088608 0.70512821 0.73076923 0.73076923] with a Mean Accuracy of 0.7143459915611815. The XGBoost 5-Fold CV Accuracies are [0.74683544 0.73417722 0.79487179 0.84615385 0.84615385]
with a Mean Accuracy of 0.7936384290814671.

Then we use the GridSearchCV algorithm to find the best hyperparameters and then use cross validation to evaluate which is the best model. Here are the resulting metrics for the best model, with the parameters decided by GridSearchCV. The Tuned Logistic Regression Metrics are Accuracy of 0.7975, Precision of 0.7647, Recall of 0.5200, F1-score of 0.6190, and Best parameters for Logistic Regression: {'C': 0.1, 'penalty': 'l2'}. The Tuned Random Forest Metrics are Accuracy of 0.7722, Precision of 0.6667, Recall of 0.5600, F1-score of 0.6087, and Best parameters for Random Forest: {'max_depth': 6}. 
Tuned KNN Metrics are Accuracy of 0.7722, Precision of 0.6842, Recall of 0.5200, F1-score of 0.5909, and Best parameters for KNN: {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'uniform'}.The Tuned GBDT Metrics are Accuracy of 0.7975, Precision of 0.7647, Recall of 0.5200, and F1-score of 0.6190. The Tuned XGBoost with Best Params Metrics are Accuracy of 0.8228, Precision of 0.7619, Recall of 0.6400, F1-score of 0.6957, with the Best parameters for XGBoost: {'max_depth': 6, 'learning_rate=0.3'}.

After comparing the accuracy measures of the different models, it appears that this Tuned XGBoost is the most optimal classifier for the diabetes dataset.


Repository contents:
In our root directory we have a Data folder, a Scripts folder, an IMG folder, a Results folder, a Vignette-Part1 RMarkdown file, a Vignette-Part1 HTML and PDF file, and the README.md file. 

In the Data folder, we have the patients' data with the 8 medical predictors and the Outcome column, and we processed the data so that any patient with missing information was removed. 

In the Scripts folder, we have the drafts of the code for both in R and in Python. We also have vignette-script.R which is a script with line annotations that replicates all results shown in the primary vignette document end-to-end. The Python code and functions are also in this folder.

In the results folder, we have the accuracy, precision, recall, and F1 score measures for the different models we used. 


In the IMG folder, we provided ROC curves graphs in order to visually show the performance of the classification models, and Confusion Matrix tables. 


The Vignette RMarkdown and HTML files are the primary Vignette document where we teach the Binary Classification method and explain how to use the different algorithms with step-by-step explanations. 


Reference list:
1. [boosted tree] (https://search.r-project.org/CRAN/refmans/parsnip/html/boost_tree.html)
2. [k-nearest neighbors] (https://www.datacamp.com/tutorial/k-nearest-neighbors-knn-classification-with-r-tutorial)
3. [XGBoost] (https://xgboost.readthedocs.io/en/stable/get_started.html)
4. [GradientBoostingClassifier] (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

