# LearningDataScience

This repository contains a collection of notebooks that explores datasets from Kaggle. These are part of my process of learning Data Science for business applications. Any feedback is welcome and appreciated =)


## List of Projects

1. Diabetes

This is a *binary classification* problem in which the aim is to predict which patients have diabetes based on a number of measurements. 

The focus of this notebook is on the *binary classification metrics*, including confusion matrix, multiple kinds of scores (accuracy, precision, recall and F1 score), ROC curve and AUC.  

There are substantial fraction of missing data that needed to be handled. These data is indicated by examples with value 0 of Glucose, BloodPressure, BMI and Insulin. Instead of removing these examples, we created dummy variables to indicate whether of not these other variables have valid values. Apart from that, no other feature engineering step was required.

A simple modeling using Logistic Regression and Random Forest is performed, without any kind of optimization.

Dataset details can be found [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database). 

2. Digits

MNIST digits dataset wwith focus on correlations between features and PCA.
