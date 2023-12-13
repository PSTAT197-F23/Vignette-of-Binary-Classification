root_dir <- rprojroot::find_rstudio_root_file()
scripts_dir <- file.path(root_dir, "scripts/python")
setwd(scripts_dir)

# install python packages
system('pip install scikit-learn pandas xgboost matplotlib')

# Run models with default parameters 
system('python3 Chris_Pt1.py')

# Get more accurate metrics using k-fold Cross Validation
system('python3 Chris_Pt2.py')

# find best hyperparameters for each model using GridSearch Cross Validation
system('python3 Chris_Pt3.py') # This takes a while!
