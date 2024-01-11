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

# Kategorik sütunları sayısal forma çevirme
from sklearn.preprocessing import LabelEncoder

# Makine öğrenmesi algoritmaları için scikit-learn'den ilgili sınıfları ekler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Model performansını değerlendirmek için confusion matrix ve ROC curve ile ilgili fonksiyonları ekler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix

# .csv dosyasını okuyoruz
df = pd.read_csv("Dataset_sdn.csv")

# Her bir sütunun içerdiği eksik verileri çubuk grafiği ile çizdirir
def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()
    missing_values.plot(kind='bar',color='DarkRed',edgecolor='black')
    plt.xlabel("Özellikler")
    plt.ylabel("Eksik değerler")
    plt.title("Toplam eksik değer")
    plt.show()

plotMissingValues(df)

# Etiket (label) sütununda kaç tane 0 ve 1 olduğunu gösteren değerleri alın
label_counts = df['label'].value_counts()

# Etiket 0 ve 1 sayılarını ekrana yazdırın
print("Etiket 0 sayısı:", label_counts[0])
print("Etiket 1 sayısı:", label_counts[1])

# 0 ve 1 etiketlerinin sayısını eşitlemek için minimum etiket sayısını bulun
min_label_count = min(label_counts[0], label_counts[1])

# 0 ve 1 etiketlerinden eşit sayıda örnek alın
balanced_df = pd.concat([
    df[df['label'] == 0].sample(min_label_count, random_state=42),
    df[df['label'] == 1].sample(min_label_count, random_state=42)
])

# Yeni veri setinin etiket sayısını kontrol edin
balanced_label_counts = balanced_df['label'].value_counts()

# Her bir özelliğin histogram grafiğini oluşturur
plt.figure(figsize=(15, 10))
for col in balanced_df.columns:
    plt.hist(balanced_df[col],color='DarkBlue',edgecolor='black')
    plt.title(col)
    plt.show()

# Sütunlardaki boşlukları temizler ve sütun isimlerini düzenler
df.columns = df.columns.str.strip()

# Null değerleri içeren satırları siler
df = df.dropna()

# DataFrame üzerinde NaN (Not a Number) değerlerini kontrol eder
pd.set_option('use_inf_as_na', True)
null_values = df.isnull().sum()

# Kategorik sütunları one-hot encoding ile çevirin
df = pd.get_dummies(df, columns=['src', 'dst'])

#Protocol sütununa labelEncoder uygulanıyor
df['Protocol'] = LabelEncoder().fit_transform(df['Protocol'])

#label etiketini ayırıp X ve y'yi belirliyoruz
X = df.drop('label', axis=1)
y = df['label']

#Normalizasyon uygulanıyor
X_normalized = StandardScaler().fit_transform(X)

#train ve test verileri bölünüyor.
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Confusion matrix çizdirme fonksiyonu
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.show()

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve, average_precision_score

# RandomUnderSampler nesnesini oluşturun
rus = RandomUnderSampler(random_state=42)

# Veriyi azaltın
X_train, y_train = rus.fit_resample(X_train, y_train)

# Verileri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#---------------- DDoS Detection ---------------------------------------------------------------------------------------

# Logistic Regression modelini oluşturur ve eğitir
lr_model = LogisticRegression(random_state=42, max_iter=250)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Logistic Regression performans metriklerini hesaplar
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)

# 5-katlı çapraz doğrulama
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)

# Çapraz doğrulama sonuçlarını yazdırma
print("Çapraz Doğrulama Sonuçları:", cv_scores)
print("Ortalama Doğruluk:", cv_scores.mean())

# Sonuçları yazdırır
print('\nLogistic Regression Metrics:')
print(f'Accuracy: {lr_accuracy:.4f}')
print(f'F1 Score: {lr_f1:.4f}')
print(f'Precision: {lr_precision:.4f}')
print(f'Recall: {lr_recall:.4f}')

# Logistic Regression için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, lr_pred, ['Normal', 'DDoS'], 'Logistic Regression Confusion Matrix')

#----------------------------------------------------------------------------------------------------------------------

# Decision Tree modelini oluşturur ve eğitir
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

# Decision Tree performans metriklerini hesaplar
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)

# 5-katlı çapraz doğrulama
cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=5)

# Çapraz doğrulama sonuçlarını yazdırma
print("Çapraz Doğrulama Sonuçları:", cv_scores)
print("Ortalama Doğruluk:", cv_scores.mean())

# Sonuçları yazdırır
print('\nDecision Tree Metrics:')
print(f'Accuracy: {dt_accuracy:.4f}')
print(f'F1 Score: {dt_f1:.4f}')
print(f'Precision: {dt_precision:.4f}')
print(f'Recall: {dt_recall:.4f}')

# Decision Tree için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, dt_pred, ['Normal', 'DDoS'], 'Decision Tree Confusion Matrix')

# Decision Tree için önemli özellikleri belirleme
dt_feature_importances = dt_model.feature_importances_
dt_feature_names = X.columns
dt_features_df = pd.DataFrame({'Feature': dt_feature_names, 'Importance': dt_feature_importances})
dt_features_df = dt_features_df.sort_values(by='Importance', ascending=False)

# Decision Tree için önemli özellikleri görselleştirme
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=dt_features_df)
plt.title('Decision Tree Feature Importance')
plt.show()

#-------------------------------------------------------------------------------------------------

# Random Forest modelini oluşturur ve eğitir
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# Random Forest performans metriklerini hesaplar
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

# 5-katlı çapraz doğrulama
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)

# Çapraz doğrulama sonuçlarını yazdırma
print("Çapraz Doğrulama Sonuçları:", cv_scores)
print("Ortalama Doğruluk:", cv_scores.mean())

# Sonuçları yazdırır
print('\nRandom Forest Metrics:')
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'F1 Score: {rf_f1:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')

# Random Forest için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, rf_pred, ['Normal', 'DDoS'], 'Random Forest Confusion Matrix')

# Random Forest
feature_importances = rf_model.feature_importances_
feature_names = X.columns
features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Random Forest Feature Importance')
plt.show()

#-------------------------------------------------------------------------------------------------

# SVM modelini oluşturur ve eğitir
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

# SVM performans metriklerini hesaplar
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)

# 5-katlı çapraz doğrulama
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)

# Çapraz doğrulama sonuçlarını yazdırma
print("Çapraz Doğrulama Sonuçları:", cv_scores)
print("Ortalama Doğruluk:", cv_scores.mean())

# Sonuçları yazdırır
print('\nSVM Metrics:')
print(f'Accuracy: {svm_accuracy:.4f}')
print(f'F1 Score: {svm_f1:.4f}')
print(f'Precision: {svm_precision:.4f}')
print(f'Recall: {svm_recall:.4f}')

# SVM için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, svm_pred, ['Normal', 'DDoS'], 'SVM Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# KNN modelini tanımla
knn_model = KNeighborsClassifier()

# Denenecek k değerleri
param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

# GridSearchCV kullanarak en iyi k değerini bulma
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# En iyi k değerini yazdırma
best_k_value = grid_search.best_params_['n_neighbors']
print(f"En iyi k değeri: {best_k_value}")

# En iyi k değeri ile modeli tekrar eğitme
best_knn_model = KNeighborsClassifier(n_neighbors=best_k_value)
best_knn_model.fit(X_train_scaled, y_train)  # En iyi k değeri ile modeli eğit

# Test seti üzerinde performansı değerlendirme
best_knn_pred = best_knn_model.predict(X_test_scaled)
best_knn_accuracy = accuracy_score(y_test, best_knn_pred)
best_knn_f1 = f1_score(y_test, best_knn_pred)
best_knn_precision = precision_score(y_test, best_knn_pred)
best_knn_recall = recall_score(y_test, best_knn_pred)

# En iyi k değeri ile modelin performansını yazdırma
print('\nEn iyi K-Nearest Neighbors Metrics:')
print(f'Accuracy: {best_knn_accuracy:.4f}')
print(f'F1 Score: {best_knn_f1:.4f}')
print(f'Precision: {best_knn_precision:.4f}')
print(f'Recall: {best_knn_recall:.4f}')

# Confusion Matrix çizdirme
plot_confusion_matrix(y_test, best_knn_pred, ['Normal', 'DDoS'], 'En iyi KNN Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# MLPClassifier modelini oluşturur ve eğitir
mlp_model = MLPClassifier(random_state=42, max_iter=300)
mlp_model.fit(X_train_scaled, y_train)

mlp_pred = mlp_model.predict(X_test_scaled)

# MLPClassifier performans metriklerini hesaplar
mlp_accuracy = accuracy_score(y_test, mlp_pred)
mlp_f1 = f1_score(y_test, mlp_pred)
mlp_precision = precision_score(y_test, mlp_pred)
mlp_recall = recall_score(y_test, mlp_pred)

# 5-katlı çapraz doğrulama
cv_scores_mlp = cross_val_score(mlp_model, X_train_scaled, y_train, cv=5)

# Çapraz doğrulama sonuçlarını yazdırma
print("MLPClassifier Çapraz Doğrulama Sonuçları:", cv_scores_mlp)
print("Ortalama Doğruluk:", cv_scores_mlp.mean())

# Sonuçları yazdırır
print('\nMLPClassifier Metrics:')
print(f'Accuracy: {mlp_accuracy:.4f}')
print(f'F1 Score: {mlp_f1:.4f}')
print(f'Precision: {mlp_precision:.4f}')
print(f'Recall: {mlp_recall:.4f}')

# MLPClassifier için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, mlp_pred, ['Normal', 'DDoS'], 'MLPClassifier Confusion Matrix')

#-------------------------------------------------------------------------------------------------

# Bernoulli Naive Bayes modelini oluşturur ve eğitir
bernoulli_nb_model = BernoulliNB()
bernoulli_nb_model.fit(X_train_scaled, y_train)
bernoulli_nb_pred = bernoulli_nb_model.predict(X_test_scaled)

# Bernoulli Naive Bayes performans metriklerini hesaplar
bernoulli_nb_accuracy = accuracy_score(y_test, bernoulli_nb_pred)
bernoulli_nb_f1 = f1_score(y_test, bernoulli_nb_pred)
bernoulli_nb_precision = precision_score(y_test, bernoulli_nb_pred)
bernoulli_nb_recall = recall_score(y_test, bernoulli_nb_pred)

# 5-katlı çapraz doğrulama
cv_scores_bernoulli_nb = cross_val_score(bernoulli_nb_model, X_train_scaled, y_train, cv=5)

# Çapraz doğrulama sonuçlarını yazdırma
print("Bernoulli Naive Bayes Çapraz Doğrulama Sonuçları:", cv_scores_bernoulli_nb)
print("Ortalama Doğruluk:", cv_scores_bernoulli_nb.mean())

# Sonuçları yazdırır
print('\nBernoulli Naive Bayes Metrics:')
print(f'Accuracy: {bernoulli_nb_accuracy:.4f}')
print(f'F1 Score: {bernoulli_nb_f1:.4f}')
print(f'Precision: {bernoulli_nb_precision:.4f}')
print(f'Recall: {bernoulli_nb_recall:.4f}')

# Bernoulli Naive Bayes için Confusion Matrix çizdirme
plot_confusion_matrix(y_test, bernoulli_nb_pred, ['Normal', 'DDoS'], 'Bernoulli Naive Bayes Confusion Matrix')

#---------------- Roc Eğrisi ---------------------------------------------------------------------------------------

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

# KNN modelinin tahmin olasılıklarını al
knn_prob = best_knn_model.predict_proba(X_test_scaled)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_prob)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Berhoulli Naive Bayes ROC Curve
nb_prob = bernoulli_nb_model.predict_proba(X_test_scaled)[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_prob)
roc_auc_nb = auc(fpr_nb, tpr_nb)

# ROC Curve çizimi
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

# Precision-Recall Curve çizimi
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





