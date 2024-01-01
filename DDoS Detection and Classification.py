# Pandas ve NumPy kütüphanelerini ekler
import pandas as pd
import numpy as np

# Grafik çizimleri için Matplotlib ve Seaborn kütüphanelerini ekler
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Veri setini eğitim ve test setlerine bölmek için scikit-learn'den ilgili fonksiyonu ekler
from sklearn.model_selection import train_test_split

# Veri standardizasyonu için scikit-learn'den ilgili fonksiyonu ekler
from sklearn.preprocessing import StandardScaler

# Makine öğrenmesi algoritmaları için scikit-learn'den ilgili sınıfları ekler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Model performansını değerlendirmek için confusion matrix ve ROC curve ile ilgili fonksiyonları ekler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

#-------------------------------------------------------------------------------------------------

# .csv dosyasını okuyoruz
df = pd.read_csv("DDoS.csv")

# Veri setinin ilk 5 satırını önizlemek için kullanılır.
df_head = df.head()

# Sütunlardaki boşlukları temizler ve sütun isimlerini düzenler
df.columns = df.columns.str.strip()

# 'Label' sütunundaki benzersiz değerleri görüntüler
unique_labels = df['Label'].unique()

#-------------------------------------------------------------------------------------------------

# Null verileri inceleyerek histogram grafiğini çizdirir
plt.figure(figsize=(10, 4))
plt.hist(df.isna().sum())
plt.xticks([0, 1], labels=['Null Degil=0', 'Null=1'])
plt.title('Kolonlara göre null degerler')
plt.xlabel('Özellik')
plt.ylabel('Null değer sayısı')
plt.show()

# Her bir sütunun içerdiği eksik verileri çubuk grafiği ile çizdirir
def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()
    fig = plt.figure(figsize=(16, 5))
    missing_values.plot(kind='bar')
    plt.xlabel("Özellikler")
    plt.ylabel("Eksik değerler")
    plt.title("Toplam eksik değer")
    plt.show()

plotMissingValues(df)

# Null değerleri içeren satırları siler
data_f = df.dropna()

# Null değerlerin tekrar kontrolünü yapar
plt.figure(figsize=(10, 4))
plt.hist(data_f.isna().sum())
plt.title('Null değerler silindikten sonra dataset')
plt.xlabel('Null değer sayısı')
plt.ylabel('Kolon sayısı')
plt.show()

# DataFrame üzerinde NaN (Not a Number) değerlerini kontrol eder
pd.set_option('use_inf_as_na', True)
null_values = data_f.isnull().sum()

# Veri tiplerini kontrol eder
(data_f.dtypes == 'object')

# 'Label' verilerini sayısal hale getirir
data_f['Label'] = data_f['Label'].map({'BENIGN': 0, 'DDoS': 1})

# BENIGN ve DDoS verilerini görselleştirir
plt.hist(data_f['Label'], bins=[0, 0.3, 0.7, 1], edgecolor='black')  
plt.xticks([0, 1], labels=['Normal=0', 'DDoS=1'])
plt.xlabel("Sınıflar")
plt.ylabel("Miktar")
plt.show()

# Veri setinin istatistiksel özetini gösterir
df.describe()

# Her bir özelliğin histogram grafiğini oluşturur
plt.figure(figsize=(15, 10))
for col in data_f.columns:
    plt.hist(data_f[col])
    plt.title(col)
    plt.show()

# 'Label' verisini ana veriden ayırır
X = data_f.drop('Label', axis=1)
y = data_f['Label']

# Eğitim ve test verilerini oluşturur
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Eğitim ve test verilerinin sayısını yazdırır
print("Train dataset size =", X_train.shape)
print("Test dataset size =", X_test.shape)

#---------------------- Sınıflandırma algoritmaları ---------------------------------------------------------------------

# Random Forest modelini oluşturur ve eğitir
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Modelin özelliklerin önem sırasını belirler
importances = rf_model.feature_importances_

# Özelliklerin önem sırasına göre sıralanması için indeksleri alır
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
feature_names = [f"Feature {i}" for i in indices]

# Özelliklerin önem sırasına göre çubuk grafiği çizdirir
plt.figure(figsize=(8, 14))
plt.barh(range(X_train.shape[1]), importances[indices], align="center")
plt.yticks(range(X_train.shape[1]), feature_names)
plt.xlabel("Önem miktarı")
plt.title("Özelliklerin önemi")
plt.show()

# Bir ağacın görsel temsilini çizdirir
from sklearn.tree import plot_tree
estimator = rf_model.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(estimator, filled=True, rounded=True)
plt.show()

# Confusion matrix çizdirme fonksiyonu
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.show()

# Random Forest performans metriklerini hesaplar ve yazdırır
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

print('\nRandom Forest Metrics:')
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'F1 Score: {rf_f1:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')

# Random Forest'in gerçek ve tahmin edilen sınıflar arasındaki detaylı ilişkiyi görselleştirir
plot_confusion_matrix(y_test, rf_pred, ['Normal', 'DDoS'], 'Random Forest Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# Logistic Regression modelini oluşturur ve eğitir
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Logistic Regression performans metriklerini hesaplar
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)

# Sonuçları yazdırır
print('\nLogistic Regression Metrics:')
print(f'Accuracy: {lr_accuracy:.4f}')
print(f'F1 Score: {lr_f1:.4f}')
print(f'Precision: {lr_precision:.4f}')
print(f'Recall: {lr_recall:.4f}')

# Logistic Regression için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, lr_pred, ['Normal', 'DDoS'], 'Logistic Regression Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# Neural Network modelini oluşturur ve eğitir
nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)

# Neural Network performans metriklerini hesaplar
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)

# Sonuçları yazdırır
print('\nNeural Network Metrics:')
print(f'Accuracy: {nn_accuracy:.4f}')
print(f'F1 Score: {nn_f1:.4f}')
print(f'Precision: {nn_precision:.4f}')
print(f'Recall: {nn_recall:.4f}')

# Neural Network için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, nn_pred, ['Normal', 'DDoS'], 'Neural Network Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# Gradient Boosting sınıflandırma modelini ekleyin
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Gradient Boosting için tahmin olasılıklarını hesaplayın
gb_proba = gb_model.predict_proba(X_test)

# Gradient Boosting sınıflandırma modelinin ROC eğrisini ve AUC değerini hesaplayın
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_proba[:, 1])
gb_auc = auc(gb_fpr, gb_tpr)

# Gradient Boosting performans metriklerini hesaplar ve yazdırır
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)

print('\nGradient Boosting Metrics:')
print(f'Accuracy: {gb_accuracy:.4f}')
print(f'F1 Score: {gb_f1:.4f}')
print(f'Precision: {gb_precision:.4f}')
print(f'Recall: {gb_recall:.4f}')

# Gradient Boosting için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, gb_pred, ['Normal', 'DDoS'], 'Gradient Boosting Confusion Matrix')
#-------------------------------------------------------------------------------------------------

# K-Nearest Neighbors (KNN) modelini ekleyin
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# KNN için tahmin olasılıklarını hesaplayın
knn_proba = knn_model.predict_proba(X_test)

# KNN sınıflandırma modelinin ROC eğrisini ve AUC değerini hesaplayın
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_proba[:, 1])
knn_auc = auc(knn_fpr, knn_tpr)

#-------------------------------------------------------------------------------------------------
# Random Forest modeli için tahmin olasılıklarını hesaplar
rf_proba = rf_model.predict_proba(X_test)

# Logistic Regression modeli için tahmin olasılıklarını hesaplar
lr_proba = lr_model.predict_proba(X_test)

# K-Nearest Neighbors (KNN) modelini oluşturur, eğitir ve tahmin olasılıklarını hesaplar
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_proba = knn_model.predict_proba(X_test)

# Neural Network modeli için tahmin olasılıklarını hesaplar
nn_proba = nn_model.predict_proba(X_test)

#-------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))

# Gerçek sınıflar
plt.subplot(1, 2, 1)
plt.hist(y_test, bins=[-0.5, 0.5, 1.5], edgecolor='black')
plt.xticks([0, 1], labels=['Normal', 'DDoS'])
plt.title('Gerçek Sınıflar')

# Tahmin edilen sınıflar
plt.subplot(1, 2, 2)
plt.hist(gb_pred, bins=[-0.5, 0.5, 1.5], edgecolor='black', color='orange')
plt.xticks([0, 1], labels=['Normal', 'DDoS'])
plt.title('Tahmin Edilen Sınıflar')

plt.show()

#-------------------------------------------------------------------------------------------------

# Combine predictions for ROC curve

#ROC (Receiver Operating Characteristic) 
#AUC(Area Under the Curve)

# Random Forest sınıflandırma modelinin ROC eğrisini ve AUC değerini hesaplar.
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba[:, 1])
rf_auc = auc(rf_fpr, rf_tpr)

# Calculate ROC curve for Logistic Regression
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba[:, 1])
lr_auc = auc(lr_fpr, lr_tpr)

# Calculate ROC curve for Neural Network
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_proba[:, 1])
nn_auc = auc(nn_fpr, nn_tpr)


# Tüm ROC eğrilerini tek bir grafik üzerinde çizdirin
plt.figure(figsize=(10, 8))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')
plt.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {gb_auc:.2f})')
plt.plot(knn_fpr, knn_tpr, label=f'K-Nearest Neighbors (AUC = {knn_auc:.2f})')

# Şans eseri model (Random Classifier) ROC eğrisi
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier (AUC = 0.50)')

plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Receiver Operating Characteristic (ROC) Eğrisi')
plt.legend()
plt.grid()
plt.show()

#-------------------------------------------------------------------------------------------------



































