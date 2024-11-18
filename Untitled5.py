#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhanced data visualizations
from sklearn.ensemble import RandomForestClassifier  # For building the Random Forest model
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report  # For evaluation metrics
from scipy.stats import zscore  # For Z-Score calculation

# Step 1: Load the dataset
# Replace 'dataset.csv' with the actual path to your dataset
data = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Step 2: Initial understanding of the dataset
print("First few rows of the dataset:")
print(data.head())  # Displays the first 5 rows of the dataset

print("\nDataset Info:")
data.info()  # Gives an overview of the dataset including null values

print("\nStatistical Summary:")
print(data.describe())  # Provides statistical details like mean, std, min, etc.

# Step 3: Data Cleaning and Preprocessing
# Checking for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())



# In[2]:


data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
data.fillna(0, inplace=True)  # Replace NaN with 0
 # Ensure all transaction amounts are non-negative
data['amount'] = data['amount'].replace(0, 1e-10)
# 2. EDA: Plot distribution of fraud and non-fraud transactions
sns.countplot(x='isFraud', data=data, palette="viridis")
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# 3. Log transformation for the amount
data['amount_log'] = np.log10(data['amount'] + 1e-6)

# Plot the log-transformed distribution
sns.histplot(data['amount_log'], bins=50, kde=True)
plt.title("Transaction Amount Distribution (Log-Transformed)")
plt.xlabel("Log10(Amount)")
plt.show()

# 4. EDA: Distribution of transaction amounts with log scale
sns.histplot(data['amount'], bins=50, kde=True, log_scale=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount (Log Scale)")
plt.ylabel("Frequency")
plt.show()

# 5. Outlier Detection with Z-Score
numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
z_scores = np.abs(zscore(data[numeric_cols]))
outliers = (z_scores > 3).any(axis=1)
data_cleaned = data[~outliers]

print(f"Outliers removed: {outliers.sum()}")
print(f"Remaining records: {data_cleaned.shape[0]}")



# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Splitting dataset into features (X) and target (y)
X = data_cleaned.drop('isFraud', axis=1)  # Features
y = data_cleaned['isFraud']  # Target variable

# Assuming 'Payment' is a categorical column in your target variable
le = LabelEncoder()
y = le.fit_transform(y)  # Encoding the target variable

# If there are missing values, you can either drop or fill them:
X.fillna(X.mean(), inplace=True)  # Imputing with mean for numerical columns
y.fillna(y.mode()[0], inplace=True)  # Imputing with mode for categorical target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 7: Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Model Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test))

# Confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 8: ROC-AUC Curve
y_probs = rf_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_probs)
print(f"\nROC-AUC Score: {auc_score}")



# In[ ]:


fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Step 9: Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)
plt.plot(recall_vals, precision_vals, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Step 10: Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', title="Top 10 Feature Importances")
plt.show()

# Step 11: Feature Importance Plot for all features
feature_importances.sort_values(ascending=False).plot(kind='barh', figsize=(10, 8), title="Feature Importances")
plt.show()

# Step 12: Correlation Heatmap (for numerical features)
correlation_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Step 13: Hyperparameter Tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
rf_model = grid_search.best_estimator_

# Step 14: Cross-validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy scores: {cv_scores}")


# In[ ]:




