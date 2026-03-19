# PRODIGY_DS_03 — Decision Tree Classifier

## 📌 Task
Build a decision tree classifier to predict whether 
a customer will purchase a product or service based 
on their demographic and behavioral data using the 
Bank Marketing Dataset from UCI.

## 🛠 Tools Used
Python | Scikit-learn | Pandas | Matplotlib | Seaborn | NumPy

## ⚙️ Model Details
- Algorithm: Decision Tree Classifier (max_depth=5)
- Dataset: Bank Marketing Dataset (41,188 records)
- Training samples: 32,950
- Testing samples: 8,238
- Encoding: Label Encoding for categorical variables

## 🎯 Model Performance
- Overall Accuracy: 91%
- No precision: 94% | Recall: 96%
- Yes precision: 65% | Recall: 54%
- Top Feature: Call Duration (importance score: 0.50)

## 📊 Key Findings
- Call duration = strongest predictor of subscription
- nr.employed (economy indicator) = 2nd most important
- Model predicts "No" very accurately (94% precision)
- Longer the call = more likely customer will subscribe!

## 📈 Visualizations Created
- plot1_decision_tree.png — Full decision tree visual
- plot2_confusion_matrix.png — Confusion matrix heatmap
- plot3_feature_importance.png — Top 10 important features
- plot4_target_distribution.png — Target class distribution
- plot5_age_subscription.png — Age vs subscription
- plot6_job_subscription.png — Job type vs subscription

## 📁 Files
- task3.py — Main code
- plot1_decision_tree.png
- plot2_confusion_matrix.png
- plot3_feature_importance.png
- plot4_target_distribution.png
- plot5_age_subscription.png
- plot6_job_subscription.png

## 🏢 Internship
Prodigy InfoTech Data Science Internship — Task 3
