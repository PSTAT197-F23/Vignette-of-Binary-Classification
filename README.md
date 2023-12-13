Vignette on implementing Binary Classification using patients' medical data in order to classify/predict if a patient has diabetes; Group 12 created as a class project for PSTAT197A in Fall 2023.

Contributors:
Group 12: Erica Chen, Chris Straw, and Claire Lee

Vignette abstract:
For our vignette class project, we decided to choose the topic Nonlinear Binary Classification, and specifically cover the binary classification algorithms: Logisitc Regression, Random Forest, Boosted Tree Model, K-Nearest Neighbors and XGBoost. Binary Classification is a method in machine learning where we can classify an observation to be under one of the two class labels based on predictors of the given data. The classes are generally labeled as 0 and 1, but can also use the labels: True and False, or positive and negative. 

The example data we used to carry out these different binary classification algorithms sourced from the National Institute of Diabetes and Digestive Kidney Disease. This dataset is patients' data of females at least twenty one years old and of Pima Indian heritage. There are eight independent variables: Pregnancies, Glucose Concentration, Blood Pressure, Skin Thickness, Insulin Levels, BMI (Body Mass Index), Diabetes Pedigree Function, and Age. The response variable is Outcome where 1 means Yes diabetes and 0 is No diabetes. Our primary objective is to develop a binary classification model that accurately predicts whether a female of Pima Indian heritage who is at least 21 years old has diabetes or not. 

We then compare the different binary classification algorithms and choose the final model based on performance metrics such as accuracy, precision, recall, and F1 score. For the Logistic Regression, the accuracy was 0.7722, the precision was 0.6957, the recall was 0.5926, and the F1 score was 0.6400. For the Random Forest, the accuracy was 0.7848, the precision was 0.7083, the recall was 0.6296, and the F1-score was 0.6667. For the K-Nearest Neighbors, the accuracy was 0.7089, the precision was 0.5769, the recall was 0.5556, and the F1-score was 0.5660. For the XGBoost, the accuracy was 0.7468, the precision was 0.6400, the recall was 0.5926, and the F1-score was 0.6154.


Repository contents:
In our root directory we have a Data folder, a Scripts folder, an IMG folder, a Vignette RMarkdown file, a Vignette HTML file, and the README.md file. 

In the Data folder, we have the raw data of the patients' data with the 8 medical predictors and the Outcome column. The processed data consists...

In the Scripts folder...
Line annotations...


In the IMG folder...

The Vignette RMarkdown and HTML files are the primary Vignette document where we teach the Binary Classification method and explain how to use the different algorithms with step-by-step explanations. 


Reference list:
1. 
2. 

