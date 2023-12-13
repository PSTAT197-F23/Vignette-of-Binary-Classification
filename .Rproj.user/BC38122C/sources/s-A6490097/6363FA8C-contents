---
title: "vignette"
output: html_document
date: "2023-12-05"
---
  
  
# load packages
library(readr)
library(vip)
library(naniar)
library(tidymodels)
library(ISLR)
library(ISLR2)
library(tidyverse)
library(glmnet)
library(modeldata)
library(ggthemes)
library(janitor)
library(kableExtra)
library(yardstick)
library(kknn)
library(corrplot)
library(themis)
library(dplyr)
library(ggplot2)
library(scales)
library(rpart.plot)
library(discrim)
library(klaR)
library(plotly)
library(ranger)
library(xgboost)
library(recipes)
library(ROSE)
library(randomForest)
tidymodels_prefer()

# read data
db <- read_csv("data/diabetes.csv")

##Data Partitioning
set.seed(3435)
db_split <- initial_split(db, prop = 0.80,
                          strata = "Outcome")
db_train <- training(db_split)
db_test <- testing(db_split)
db_fold <- vfold_cv(db_train,v=4)

#Recipe
db_train$Outcome <- factor(db_train$Outcome)
db_recipe_demo <- recipe(Outcome ~ ., data = db_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>% 
  step_upsample(Outcome, over_ratio = 0.5, skip = FALSE)

prep(db_recipe_demo) %>% bake(new_data = db_train) %>% 
  group_by(Outcome) %>% 
  summarise(count = n())

db_recipe <- recipe(Outcome ~ ., data = db_train ) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>% 
  step_upsample(Outcome, over_ratio = 1)


#Binary Classification Algorithm #1 
#Logistic Regression

log_reg <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

db_log_wflow <- workflow() %>% 
  add_model(log_reg) %>% 
  add_recipe(db_recipe)

# Tune the model
db_tune_reg <- tune_grid(
  object = db_log_wflow, 
  resamples = db_fold
)


# Fit the model
log_fit <- fit(db_log_wflow, db_train)

# Extract predictions
log_preds <- augment(log_fit, new_data = db_train)

# Calculate ROC AUC directly using pROC
roc_curve <- roc(log_preds$Outcome, log_preds$.pred_0)
roc_auc_value <- auc(roc_curve)
print(roc_auc_value)

# Confusion Matrix
conf_matrix <- log_preds %>%
  conf_mat(truth = Outcome, estimate = .pred_class)

# Plot Confusion Matrix
conf_matrix %>% autoplot(type = 'heatmap') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

# Show best tuning parameters
show_best(db_tune_reg, n = 1)


# Accuracy Measures
# Extract predictions
log_preds <- augment(log_fit, new_data = db_train)

log_predictions <- augment(log_fit, new_data = db_test)
log_accuracy <- accuracy(log_predictions, truth = Outcome, estimate = .pred_class)
log_conf_matrix <- conf_mat(log_predictions, truth = Outcome, estimate = .pred_class)


# Calculate ROC AUC directly using pROC
roc_curve <- roc(log_preds$Outcome, log_preds$.pred_0)
roc_auc_value <- auc(roc_curve)
print(roc_auc_value)

# Confusion Matrix
conf_matrix <- log_preds %>%
  conf_mat(truth = Outcome, estimate = .pred_class)

# Plot Confusion Matrix
conf_matrix %>% autoplot(type = 'heatmap') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

# Show best tuning parameters
show_best(db_tune_reg, n = 1)


#Binary Classification Algorithm #3
#K-Nearest Neighbors

# Recipe without step_upsample
db_recipe <- recipe(Outcome ~ ., data = db_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors())

# Prepare the recipe
prep(db_recipe)%>% bake(new_data = db_train) %>% 
  group_by(Outcome) %>% 
  summarise(count = n())



# Define the k-NN model
knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

# Create a workflow with the recipe and model
db_knn_wflow <- workflow() %>% 
  add_recipe(db_recipe) %>%  # Include the recipe
  add_model(knn_model)      # Include the model

# Grid for tuning
neighbors_grid <- grid_regular(neighbors(range = c(1, 10)), levels = 10)

# Tune the model
db_tune_knn <- tune_grid(
  object = db_knn_wflow, 
  resamples = db_fold, 
  grid = neighbors_grid
)

# Select the best model
best_knn_db <- select_best(
  db_tune_knn,
  metric = "roc_auc",
  neighbors
)


# We will then use the predictions and observed classes to create a Confusion Matrix table.

best_knn_wf <- finalize_workflow(db_knn_wflow, best_knn_db)
knn_fit <- fit(best_knn_wf, data = db_train)

# Extract predictions
knn_preds <- augment(knn_fit, new_data = db_train)

# Calculate ROC AUC directly using pROC
roc_curve <- roc(knn_preds$Outcome, knn_preds$.pred_0)
roc_auc_value <- auc(roc_curve)
print(roc_auc_value)

# Confusion Matrix
conf_matrix <- knn_preds %>%
  conf_mat(truth = Outcome, estimate = .pred_class)

# Plot Confusion Matrix
conf_matrix %>% autoplot(type = 'heatmap') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

autoplot(db_tune_knn) + theme_minimal()



#Binary Classification Algorithm #5
#Boosted Tree Model

bt_class_spec <- boost_tree(mtry = tune(), 
                            trees = tune(), 
                            learn_rate = tune()) %>%
  set_engine("xgboost") %>% 
  set_mode("classification")

bt_class_wf <- workflow() %>% 
  add_model(bt_class_spec) %>% 
  add_recipe(db_recipe)

bt_grid <- grid_regular(mtry(range = c(1, 6)), 
                        trees(range = c(200, 600)),
                        learn_rate(range = c(-10, -1)),
                        levels = 5)
bt_grid


tune_bt_class <- tune_grid(
  bt_class_wf,
  resamples = db_fold,
  grid = bt_grid
)
save(tune_bt_class, file = "tune_bt_class.rda")

load("tune_bt_class.rda")
autoplot(tune_bt_class) + theme_minimal()



show_best(tune_bt_class, n = 1)



best_bt_class <- select_best(tune_bt_class)


bt_mode_fit <- finalize_workflow(bt_class_wf, best_bt_class)
bt_mode_fit <- fit(bt_mode_fit, db_train)

bt_mode_fit %>% extract_fit_parsnip() %>% 
  vip() +
  theme_minimal()


# Compute the accuracy measures and create a confusion matrix
print(paste("Boosted Trees AUC:", bt_auc_value))
print("Boosted Trees Confusion Matrix:")
print(bt_conf_matrix)


# all Python code below
library(reticulate)
root_dir <- rprojroot::find_rstudio_root_file()
scripts_dir <- file.path(root_dir, "scripts/python")
setwd(scripts_dir)

# Run models with default parameters 
system('python3 Chris_Pt1.py')

# Get more accurate metrics using k-fold Cross Validation
system('python3 Chris_Pt2.py')

# find best hyperparameters for each model using GridSearch Cross Validation
system('python3 Chris_Pt3.py') # This takes a while!

# Random Forest, Gradient Boosting Decision Trees, XGBoost 

# load packages

#import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
#import matplotlib.pyplot as plt

# Load the dataset
#db = pd.read_csv("../../data/diabetes.csv")

# Process data
# Remove rows where any column except 'Pregnancies' or 'Outcome' is 0
#cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#db_filtered = db[(db[cols_to_check] != 0).all(axis=1)]

# Separate features and target variable
#X = db_filtered.drop('Outcome', axis=1)
#y = db_filtered['Outcome']

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Train Random Forest Classifier
#rf_model = RandomForestClassifier(random_state=0)
#rf_model.fit(X_train, y_train)
#compute_metrics_and_plot(rf_model, X_test, y_test, "Random Forest","RanF")

# Train Gradient Boosting Decision Trees (GBDT)
#gbdt_model = GradientBoostingClassifier(random_state=0)
#gbdt_model.fit(X_train, y_train)
#compute_metrics_and_plot(gbdt_model, X_test, y_test, "Gradient Boosted Trees","GBDT")


# Train XGBoost Classifier
#xgb_model = XGBClassifier(random_state=0,learning_rate=0.3)
#xgb_model.fit(X_train, y_train)
#compute_metrics_and_plot(xgb_model, X_test, y_test, "XGBoost","XGB")



