# Add Pandas and NumPy libraries
import pandas as pd
import numpy as np

# Add Matplotlib and Seaborn libraries for plotting graphs
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Add relevant function from scikit-learn to split data into training and test sets
from sklearn.model_selection import train_test_split

# Add relevant function from scikit-learn for data standardization
from sklearn.preprocessing import StandardScaler

# Convert categorical columns to numerical form
from sklearn.preprocessing import LabelEncoder

# Add relevant classes from scikit-learn for machine learning algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Add functions related to confusion matrix and ROC curve to evaluate model performance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix

# Read the .csv file
df = pd.read_csv("Dataset_sdn.csv")

# Plot missing values in each column as a bar chart
def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()
    missing_values.plot(kind='bar',color='DarkRed',edgecolor='black')
    plt.xlabel("Features")
    plt.ylabel("Missing values")
    plt.title("Total missing values")
    plt.show()

plotMissingValues(df)

# Get values showing how many 0s and 1s are in the label column
label_counts = df['label'].value_counts()

# Print the number of label 0s and 1s
print("Number of label 0:", label_counts[0])
print("Number of label 1:", label_counts[1])

# Find minimum label count to equalize number of 0 and 1 labels
min_label_count = min(label_counts[0], label_counts[1])

# Take equal number of samples from 0 and 1 labels
balanced_df = pd.concat([
    df[df['label'] == 0].sample(min_label_count, random_state=42),
    df[df['label'] == 1].sample(min_label_count, random_state=42)
])

# Check label count of new dataset
balanced_label_counts = balanced_df['label'].value_counts()

# Create histogram plot for each feature
plt.figure(figsize=(15, 10))
for col in balanced_df.columns:
    plt.hist(balanced_df[col],color='DarkBlue',edgecolor='black')
    plt.title(col)
    plt.show()

# Clean spaces in columns and organize column names
df.columns = df.columns.str.strip()

# Delete rows containing null values
df = df.dropna()

# Check for NaN (Not a Number) values in DataFrame
pd.set_option('use_inf_as_na', True)
null_values = df.isnull().sum()

# Convert categorical columns with one-hot encoding
df = pd.get_dummies(df, columns=['src', 'dst'])

# Apply labelEncoder to Protocol column
df['Protocol'] = LabelEncoder().fit_transform(df['Protocol'])

# Separate label and determine X and y
X = df.drop('label', axis=1)
y = df['label']

# Apply normalization
X_normalized = StandardScaler().fit_transform(X)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve, average_precision_score

# Create RandomUnderSampler object
rus = RandomUnderSampler(random_state=42)

# Reduce data
X_train, y_train = rus.fit_resample(X_train, y_train)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#---------------- DDoS Detection ---------------------------------------------------------------------------------------

# Create and train Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=250)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Calculate Logistic Regression performance metrics
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)

# 5-fold cross validation
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)

# Print cross validation results
print("Cross Validation Results:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Print results
print('\nLogistic Regression Metrics:')
print(f'Accuracy: {lr_accuracy:.4f}')
print(f'F1 Score: {lr_f1:.4f}')
print(f'Precision: {lr_precision:.4f}')
print(f'Recall: {lr_recall:.4f}')

# Plot Confusion Matrix for Logistic Regression
plot_confusion_matrix(y_test, lr_pred, ['Normal', 'DDoS'], 'Logistic Regression Confusion Matrix')

#----------------------------------------------------------------------------------------------------------------------

# Create and train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

# Calculate Decision Tree performance metrics
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)

# 5-fold cross validation
cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=5)

# Print cross validation results
print("Cross Validation Results:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Print results
print('\nDecision Tree Metrics:')
print(f'Accuracy: {dt_accuracy:.4f}')
print(f'F1 Score: {dt_f1:.4f}')
print(f'Precision: {dt_precision:.4f}')
print(f'Recall: {dt_recall:.4f}')

# Plot Confusion Matrix for Decision Tree
plot_confusion_matrix(y_test, dt_pred, ['Normal', 'DDoS'], 'Decision Tree Confusion Matrix')

# Determine important features for Decision Tree
dt_feature_importances = dt_model.feature_importances_
dt_feature_names = X.columns
dt_features_df = pd.DataFrame({'Feature': dt_feature_names, 'Importance': dt_feature_importances})
dt_features_df = dt_features_df.sort_values(by='Importance', ascending=False)

# Visualize important features for Decision Tree
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=dt_features_df)
plt.title('Decision Tree Feature Importance')
plt.show()

#-------------------------------------------------------------------------------------------------

# Create and train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# Calculate Random Forest performance metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

# 5-fold cross validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)

# Print cross validation results
print("Cross Validation Results:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Print results
print('\nRandom Forest Metrics:')
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'F1 Score: {rf_f1:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')

# Plot Confusion Matrix for Random Forest
plot_confusion_matrix(y_test, rf_pred, ['Normal', 'DDoS'], 'Random Forest Confusion Matrix')

# Random Forest feature importance
feature_importances = rf_model.feature_importances_
feature_names = X.columns
features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Random Forest Feature Importance')
plt.show()

#-------------------------------------------------------------------------------------------------

# Create and train SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

# Calculate SVM performance metrics
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)

# 5-fold cross validation
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)

# Print cross validation results
print("Cross Validation Results:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Print results
print('\nSVM Metrics:')
print(f'Accuracy: {svm_accuracy:.4f}')
print(f'F1 Score: {svm_f1:.4f}')
print(f'Precision: {svm_precision:.4f}')
print(f'Recall: {svm_recall:.4f}')

# Plot Confusion Matrix for SVM
plot_confusion_matrix(y_test, svm_pred, ['Normal', 'DDoS'], 'SVM Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# Define KNN model
knn_model = KNeighborsClassifier()

# k values to try
param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

# Find best k value using GridSearchCV
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Print best k value
best_k_value = grid_search.best_params_['n_neighbors']
print(f"Best k value: {best_k_value}")

# Retrain model with best k value
best_knn_model = KNeighborsClassifier(n_neighbors=best_k_value)
best_knn_model.fit(X_train_scaled, y_train)  # Train model with best k value

# Evaluate performance on test set
best_knn_pred = best_knn_model.predict(X_test_scaled)
best_knn_accuracy = accuracy_score(y_test, best_knn_pred)
best_knn_f1 = f1_score(y_test, best_knn_pred)
best_knn_precision = precision_score(y_test, best_knn_pred)
best_knn_recall = recall_score(y_test, best_knn_pred)

# Print performance of model with best k value
print('\nBest K-Nearest Neighbors Metrics:')
print(f'Accuracy: {best_knn_accuracy:.4f}')
print(f'F1 Score: {best_knn_f1:.4f}')
print(f'Precision: {best_knn_precision:.4f}')
print(f'Recall: {best_knn_recall:.4f}')

# Plot Confusion Matrix
plot_confusion_matrix(y_test, best_knn_pred, ['Normal', 'DDoS'], 'Best KNN Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# Create and train MLPClassifier model
mlp_model = MLPClassifier(random_state=42, max_iter=300)
mlp_model.fit(X_train_scaled, y_train)

mlp_pred = mlp_model.predict(X_test_scaled)

# Calculate MLPClassifier performance metrics
mlp_accuracy = accuracy_score(y_test, mlp_pred)
mlp_f1 = f1_score(y_test, mlp_pred)
mlp_precision = precision_score(y_test, mlp_pred)
mlp_recall = recall_score(y_test, mlp_pred)

# 5-fold cross validation
cv_scores_mlp = cross_val_score(mlp_model, X_train_scaled, y_train, cv=5)

# Print cross validation results
print("MLPClassifier Cross Validation Results:", cv_scores_mlp)
print("Mean Accuracy:", cv_scores_mlp.mean())

# Print results
print('\nMLPClassifier Metrics:')
print(f'Accuracy: {mlp_accuracy:.4f}')
print(f'F1 Score: {mlp_f1:.4f}')
print(f'Precision: {mlp_precision:.4f}')
print(f'Recall: {mlp_recall:.4f}')

# Plot Confusion Matrix for MLPClassifier
plot_confusion_matrix(y_test, mlp_pred, ['Normal', 'DDoS'], 'MLPClassifier Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# Create and train Bernoulli Naive Bayes model
bernoulli_nb_model = BernoulliNB()
bernoulli_nb_model.fit(X_train_scaled, y_train)
bernoulli_nb_pred = bernoulli_nb_model.predict(X_test_scaled)

# Calculate Bernoulli Naive Bayes performance metrics
bernoulli_nb_accuracy = accuracy_score(y_test, bernoulli_nb_pred)
bernoulli_nb_f1 = f1_score(y_test, bernoulli_nb_pred)
bernoulli_nb_precision = precision_score(y_test, bernoulli_nb_pred)
bernoulli_nb_recall = recall_score(y_test, bernoulli_nb_pred)

# 5-fold cross validation
cv_scores_bernoulli_nb = cross_val_score(bernoulli_nb_model, X_train_scaled, y_train, cv=5)

# Print cross validation results
print("Bernoulli Naive Bayes Cross Validation Results:", cv_scores_bernoulli_nb)
print("Mean Accuracy:", cv_scores_bernoulli_nb.mean())

# Print results
print('\nBernoulli Naive Bayes Metrics:')
print(f'Accuracy: {bernoulli_nb_accuracy:.4f}')
print(f'F1 Score: {bernoulli_nb_f1:.4f}')
print(f'Precision: {bernoulli_nb_precision:.4f}')
print(f'Recall: {bernoulli_nb_recall:.4f}')

# Plot Confusion Matrix for Bernoulli Naive Bayes
plot_confusion_matrix(y_test, bernoulli_nb_pred, ['Normal', 'DDoS'], 'Bernoulli Naive Bayes Confusion Matrix')

#---------------- ROC Curve ---------------------------------------------------------------------------------------

# Logistic Regression ROC Curve
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Decision Tree ROC Curve
dt_prob = dt_model.predict_proba(X_test_scaled)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# Random Forest ROC Curve
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# SVM ROC Curve
svm_prob = svm_model.decision_function(X_test_scaled)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Get prediction probabilities for KNN model
knn_prob = best_knn_model.predict_proba(X_test_scaled)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_prob)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Berhoulli Naive Bayes ROC Curve
nb_prob = bernoulli_nb_model.predict_proba(X_test_scaled)[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_prob)
roc_auc_nb = auc(fpr_nb, tpr_nb)

# Plot ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Gaussian Naive Bayes (AUC = {roc_auc_nb:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#-------------------------------------------------------------------------------------------------

# Logistic Regression
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_prob)
lr_avg_precision = average_precision_score(y_test, lr_prob)

# Decision Tree
dt_precision, dt_recall, _ = precision_recall_curve(y_test, dt_prob)
dt_avg_precision = average_precision_score(y_test, dt_prob)

# Random Forest
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_prob)
rf_avg_precision = average_precision_score(y_test, rf_prob)

# SVM
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_prob)
svm_avg_precision = average_precision_score(y_test, svm_prob)

# KNN
knn_precision, knn_recall, _ = precision_recall_curve(y_test, knn_prob)
knn_avg_precision = average_precision_score(y_test, knn_prob)

# Gaussian Naive Bayes
nb_precision, nb_recall, _ = precision_recall_curve(y_test, nb_prob)
nb_avg_precision = average_precision_score(y_test, nb_prob)

# Plot Precision-Recall Curve
plt.figure(figsize=(10, 8))

# Logistic Regression
plt.plot(lr_recall, lr_precision, color='b', label=f'Logistic Regression (AP = {lr_avg_precision:.2f})')

# Decision Tree
plt.plot(dt_recall, dt_precision, color='g', label=f'Decision Tree (AP = {dt_avg_precision:.2f})')

# Random Forest
plt.plot(rf_recall, rf_precision, color='r', label=f'Random Forest (AP = {rf_avg_precision:.2f})')

# SVM
plt.plot(svm_recall, svm_precision, color='c', label=f'SVM (AP = {svm_avg_precision:.2f})')

# KNN
plt.plot(knn_recall, knn_precision, color='m', label=f'KNN (AP = {knn_avg_precision:.2f})')

# Gaussian Naive Bayes
plt.plot(nb_recall, nb_precision, color='y', label=f'Gaussian Naive Bayes (AP = {nb_avg_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.show()

#-------------------------------------------------------------------------------------------------
