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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import plot_tree

# Add functions related to confusion matrix and ROC curve to evaluate model performance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Read the .csv file
df = pd.read_csv("Classification.csv")

# Clean spaces in columns and organize column names
df.columns = df.columns.str.strip()

# Display unique values in the 'Label' column
unique_labels = df['Label'].unique()

# Plot bar graph showing missing data in each column
def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()
    fig = plt.figure(figsize=(16, 5))
    missing_values.plot(kind='bar',color='DarkRed',edgecolor='black')
    plt.xlabel("Features")
    plt.ylabel("Missing values")
    plt.title("Total missing values")
    plt.show()

plotMissingValues(df)

# Delete rows containing null values
df = df.dropna()

# Check null values again
plt.figure(figsize=(10, 4))
plt.hist(df.isna().sum(), color='darkred', edgecolor='black')
plt.title('Dataset after removing null values')
plt.xlabel('Number of null values')
plt.ylabel('Number of columns')
plt.show()

# Check NaN (Not a Number) values in DataFrame
pd.set_option('use_inf_as_na', True)
null_values = df.isnull().sum()

# Check data types
(df.dtypes == 'object')

# Combine UDP and UDPLag classes
df['Label'] = df['Label'].replace({'UDP': 'UDP', 'UDP-lag': 'UDP'})

# Combine classes starting with 'DrDoS_' prefix as 'DrDoS'
df['Label'] = df['Label'].replace({'DrDoS_DNS': 'DrDoS', 'DrDoS_LDAP': 'DrDoS', 'DrDoS_MSSQL': 'DrDoS',
                                   'DrDoS_NetBIOS': 'DrDoS', 'DrDoS_NTP': 'DrDoS', 'DrDoS_SNMP': 'DrDoS',
                                   'DrDoS_SSDP': 'DrDoS', 'DrDoS_UDP': 'DrDoS'})

# Find unique values and their counts in 'Label' column
label_counts = df['Label'].value_counts()

# Print unique values and their counts
print("Unique Label Values and Counts:")
print(label_counts)

# Class names and counts
classes = [
    'NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Portmap', 'Syn', 'UDP', 'UDPLag',
    'DrDoS', 'TFTP', 'WebDDoS'
]

# Assign different values to each class
class_mapping = {
    
    'BENIGN': 0,
    'NetBIOS': 1,
    'LDAP': 2,
    'MSSQL': 3,
    'Portmap': 4,
    'Syn': 5,
    'UDP': 6,
    'UDPLag': 7,
    'WebDDoS': 8,
    'TFTP': 9,
    'DrDoS': 10
}

# Convert 'Label' data to numerical form
df['Label'] = df['Label'].map(class_mapping)

# Create bar graph
plt.bar(classes, 19, color='DarkRed', align='center', edgecolor='black')
plt.xlabel("Classes")
plt.ylabel("Amount")
plt.title("Class Distribution")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Show statistical summary of dataset
df.describe()

df.drop(['Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP'], axis=1, inplace=True)

le = LabelEncoder()
df['Protocol'] = le.fit_transform(df['Protocol'])

categorical_cols = ['Protocol']
df = pd.get_dummies(df, columns=categorical_cols)

# Clean spaces in columns and organize column names
df.columns = df.columns.str.strip()

# Delete rows containing null values
df = df.dropna()

# Check null values again
plt.figure(figsize=(10, 4))
plt.hist(df.isna().sum(), color='darkred', edgecolor='black')
plt.title('Dataset after removing null values')
plt.xlabel('Number of null values')
plt.ylabel('Number of columns')
plt.show()

# Check NaN (Not a Number) values in DataFrame
pd.set_option('use_inf_as_na', True)
null_values = df.isnull().sum()

# Filter rows in 'Label' column
df = df[df['Label'] != 4]
df = df[df['Label'] != 7]
df = df[df['Label'] != 8]
df = df[df['Label'] != 9]
df = df[df['Label'] != 10]

# Check data types
(df.dtypes == 'object')

X = df.drop('Label', axis=1)
y = df['Label']

df = df.astype(float)

# Create histogram for each feature
plt.figure(figsize=(15, 10))
for col in df.columns:
    plt.hist(df[col],color='DarkRed',edgecolor='black')
    plt.title(col)
    plt.show()

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Create training and test data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.40, random_state=42)

# Print training and test data sizes
print("Train dataset size =", X_train.shape)
print("Test dataset size =", X_test.shape)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#------------------------------------------------

# Custom confusion matrix plotting
def plot_confusion_matrix_custom(y_true, y_pred, classes, title, cmap='Reds'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#------------------------------------------------

# Create and train Neural Network model
nn_model = MLPClassifier(hidden_layer_sizes=(25,), max_iter=1000, random_state=500)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)

# Calculate Neural Network performance metrics
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred, average='weighted')
nn_precision = precision_score(y_test, nn_pred, average='weighted', zero_division=1)
nn_recall = recall_score(y_test, nn_pred, average='weighted', zero_division=1)

# Print results
print('\nNeural Network Metrics:')
print(f'Accuracy: {nn_accuracy:.4f}')
print(f'F1 Score: {nn_f1:.4f}')
print(f'Precision: {nn_precision:.4f}')
print(f'Recall: {nn_recall:.4f}')

# Plot custom confusion matrix for Neural Network
plot_confusion_matrix_custom(y_test, nn_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Neural Network Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Create and train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=500)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Calculate Decision Tree performance metrics
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average='weighted')
dt_precision = precision_score(y_test, dt_pred, average='weighted', zero_division=1)
dt_recall = recall_score(y_test, dt_pred, average='weighted', zero_division=1)

# Print results
print('\nDecision Tree Metrics:')
print(f'Accuracy: {dt_accuracy:.4f}')
print(f'F1 Score: {dt_f1:.4f}')
print(f'Precision: {dt_precision:.4f}')
print(f'Recall: {dt_recall:.4f}')

# Plot custom confusion matrix for Decision Tree
plot_confusion_matrix_custom(y_test, dt_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Decision Tree Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Create and train Random Forest model
rf_model = RandomForestClassifier(random_state=500)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Calculate Random Forest performance metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=1)
rf_recall = recall_score(y_test, rf_pred, average='weighted', zero_division=1)

# Print results
print('\nRandom Forest Metrics:')
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'F1 Score: {rf_f1:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')

# Plot custom confusion matrix for Random Forest
plot_confusion_matrix_custom(y_test, rf_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Random Forest Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Create and train Support Vector Machine model
svm_model = SVC(random_state=500)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Calculate Support Vector Machine performance metrics
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
svm_precision = precision_score(y_test, svm_pred, average='weighted', zero_division=1)
svm_recall = recall_score(y_test, svm_pred, average='weighted', zero_division=1)

# Print results
print('\nSupport Vector Machine Metrics:')
print(f'Accuracy: {svm_accuracy:.4f}')
print(f'F1 Score: {svm_f1:.4f}')
print(f'Precision: {svm_precision:.4f}')
print(f'Recall: {svm_recall:.4f}')

# Plot custom confusion matrix for Support Vector Machine
plot_confusion_matrix_custom(y_test, svm_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Support Vector Machine Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Create and train Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=500)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Calculate Gradient Boosting performance metrics
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred, average='weighted')
gb_precision = precision_score(y_test, gb_pred, average='weighted', zero_division=1)
gb_recall = recall_score(y_test, gb_pred, average='weighted', zero_division=1)

# Print results
print('\nGradient Boosting Metrics:')
print(f'Accuracy: {gb_accuracy:.4f}')
print(f'F1 Score: {gb_f1:.4f}')
print(f'Precision: {gb_precision:.4f}')
print(f'Recall: {gb_recall:.4f}')

# Plot custom confusion matrix for Gradient Boosting
plot_confusion_matrix_custom(y_test, gb_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Gradient Boosting Confusion Matrix', cmap='Reds')

#------------------------------------------------

# ROC curve plotting for each model
plt.figure(figsize=(10, 8))

# Neural Network ROC
nn_prob = nn_model.predict_proba(X_test)
nn_prob_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 5, 6])
nn_fpr, nn_tpr, _ = roc_curve(nn_prob_bin.ravel(), nn_prob[:, 1])
nn_auc = auc(nn_fpr, nn_tpr)
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')

# Decision Tree ROC
dt_prob = dt_model.predict_proba(X_test)
dt_prob_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 5, 6])
dt_fpr, dt_tpr, _ = roc_curve(dt_prob_bin.ravel(), dt_prob[:, 1])
dt_auc = auc(dt_fpr, dt_tpr)
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.2f})')

# Random Forest ROC
rf_prob = rf_model.predict_proba(X_test)
rf_prob_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 5, 6])
rf_fpr, rf_tpr, _ = roc_curve(rf_prob_bin.ravel(), rf_prob[:, 1])
rf_auc = auc(rf_fpr, rf_tpr)
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')

# Support Vector Machine ROC
svm_prob = svm_model.decision_function(X_test)
svm_prob_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 5, 6])
svm_fpr, svm_tpr, _ = roc_curve(svm_prob_bin.ravel(), svm_prob)
svm_auc = auc(svm_fpr, svm_tpr)
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})')

# Gradient Boosting ROC
gb_prob = gb_model.predict_proba(X_test)
gb_prob_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 5, 6])
gb_fpr, gb_tpr, _ = roc_curve(gb_prob_bin.ravel(), gb_prob[:, 1])
gb_auc = auc(gb_fpr, gb_tpr)
plt.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {gb_auc:.2f})')

# Other adjustments
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve')
plt.legend()
plt.show()
