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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import plot_tree

# Model performansını değerlendirmek için confusion matrix ve ROC curve ile ilgili fonksiyonları ekler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle

# .csv dosyasını okuyoruz
df = pd.read_csv("Classification.csv")

# Sütunlardaki boşlukları temizler ve sütun isimlerini düzenler
df.columns = df.columns.str.strip()

# 'Label' sütunundaki benzersiz değerleri görüntüler
unique_labels = df['Label'].unique()

# Her bir sütunun içerdiği eksik verileri çubuk grafiği ile çizdirir
def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()
    fig = plt.figure(figsize=(16, 5))
    missing_values.plot(kind='bar',color='DarkRed',edgecolor='black')
    plt.xlabel("Özellikler")
    plt.ylabel("Eksik değerler")
    plt.title("Toplam eksik değer")
    plt.show()

plotMissingValues(df)

# Null değerleri içeren satırları siler
df = df.dropna()

# Null değerlerin tekrar kontrolünü yapar
plt.figure(figsize=(10, 4))
plt.hist(df.isna().sum(), color='darkred', edgecolor='black')
plt.title('Null değerler silindikten sonra dataset')
plt.xlabel('Null değer sayısı')
plt.ylabel('Kolon sayısı')
plt.show()

# DataFrame üzerinde NaN (Not a Number) değerlerini kontrol eder
pd.set_option('use_inf_as_na', True)
null_values = df.isnull().sum()

# Veri tiplerini kontrol eder
(df.dtypes == 'object')

# UDP ve UDPLag sınıflarını birleştir
df['Label'] = df['Label'].replace({'UDP': 'UDP', 'UDP-lag': 'UDP'})

# 'DrDoS_' ön eki ile başlayan sınıfları 'DrDoS' olarak birleştir
df['Label'] = df['Label'].replace({'DrDoS_DNS': 'DrDoS', 'DrDoS_LDAP': 'DrDoS', 'DrDoS_MSSQL': 'DrDoS',
                                   'DrDoS_NetBIOS': 'DrDoS', 'DrDoS_NTP': 'DrDoS', 'DrDoS_SNMP': 'DrDoS',
                                   'DrDoS_SSDP': 'DrDoS', 'DrDoS_UDP': 'DrDoS'})

# 'Label' sütunundaki benzersiz değerleri ve sayılarını bulma
label_counts = df['Label'].value_counts()

# Benzersiz değerleri ve sayılarını ekrana yazdırma
print("Benzersiz Etiket Değerleri ve Sayıları:")
print(label_counts)

# Sınıf adları ve sayıları
classes = [
    'NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Portmap', 'Syn', 'UDP', 'UDPLag',
    'DrDoS', 'TFTP', 'WebDDoS'
]

# Her sınıfa farklı değer atama
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

# 'Label' verilerini sayısal hale getirir
df['Label'] = df['Label'].map(class_mapping)

# Çubuk grafik oluştur
plt.bar(classes, 19, color='DarkRed', align='center', edgecolor='black')
plt.xlabel("Sınıflar")
plt.ylabel("Miktar")
plt.title("Sınıf Dağılımı")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Veri setinin istatistiksel özetini gösterir
df.describe()

df.drop(['Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP'], axis=1, inplace=True)

le = LabelEncoder()
df['Protocol'] = le.fit_transform(df['Protocol'])

categorical_cols = ['Protocol']
df = pd.get_dummies(df, columns=categorical_cols)

# Sütunlardaki boşlukları temizler ve sütun isimlerini düzenler
df.columns = df.columns.str.strip()

# Null değerleri içeren satırları siler
df = df.dropna()

# Null değerlerin tekrar kontrolünü yapar
plt.figure(figsize=(10, 4))
plt.hist(df.isna().sum(), color='darkred', edgecolor='black')
plt.title('Null değerler silindikten sonra dataset')
plt.xlabel('Null değer sayısı')
plt.ylabel('Kolon sayısı')
plt.show()

# DataFrame üzerinde NaN (Not a Number) değerlerini kontrol eder
pd.set_option('use_inf_as_na', True)
null_values = df.isnull().sum()

# 'Label' sütunundaki satırları filtrele
df = df[df['Label'] != 4]
df = df[df['Label'] != 7]
df = df[df['Label'] != 8]
df = df[df['Label'] != 9]
df = df[df['Label'] != 10]

# Veri tiplerini kontrol eder
(df.dtypes == 'object')

X = df.drop('Label', axis=1)
y = df['Label']

df = df.astype(float)

# Her bir özelliğin histogram grafiğini oluşturur
plt.figure(figsize=(15, 10))
for col in df.columns:
    plt.hist(df[col],color='DarkRed',edgecolor='black')
    plt.title(col)
    plt.show()

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Eğitim ve test verilerini oluşturur
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.40, random_state=42)

# Eğitim ve test verilerinin sayısını yazdırır
print("Train dataset size =", X_train.shape)
print("Test dataset size =", X_test.shape)

# Verileri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#------------------------------------------------

# Özel confusion matrix çizimi
def plot_confusion_matrix_custom(y_true, y_pred, classes, title, cmap='Reds'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#------------------------------------------------

# Neural Network modelini oluşturur ve eğitir
nn_model = MLPClassifier(hidden_layer_sizes=(25,), max_iter=1000, random_state=500)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)

# Neural Network performans metriklerini hesaplar
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred, average='weighted')
nn_precision = precision_score(y_test, nn_pred, average='weighted', zero_division=1)
nn_recall = recall_score(y_test, nn_pred, average='weighted', zero_division=1)

# Sonuçları yazdırır
print('\nNeural Network Metrics:')
print(f'Accuracy: {nn_accuracy:.4f}')
print(f'F1 Score: {nn_f1:.4f}')
print(f'Precision: {nn_precision:.4f}')
print(f'Recall: {nn_recall:.4f}')

# Neural Network için özel confusion matrix çizdirme
plot_confusion_matrix_custom(y_test, nn_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Neural Network Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Decision Tree modelini oluşturur ve eğitir
dt_model = DecisionTreeClassifier(random_state=500)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Decision Tree performans metriklerini hesaplar
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average='weighted')
dt_precision = precision_score(y_test, dt_pred, average='weighted', zero_division=1)
dt_recall = recall_score(y_test, dt_pred, average='weighted', zero_division=1)

# Sonuçları yazdırır
print('\nDecision Tree Metrics:')
print(f'Accuracy: {dt_accuracy:.4f}')
print(f'F1 Score: {dt_f1:.4f}')
print(f'Precision: {dt_precision:.4f}')
print(f'Recall: {dt_recall:.4f}')

# Decision Tree için özel confusion matrix çizdirme
plot_confusion_matrix_custom(y_test, dt_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Decision Tree Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Random Forest modelini oluşturur ve eğitir
rf_model = RandomForestClassifier(random_state=500)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Random Forest performans metriklerini hesaplar
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=1)
rf_recall = recall_score(y_test, rf_pred, average='weighted', zero_division=1)

# Sonuçları yazdırır
print('\nRandom Forest Metrics:')
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'F1 Score: {rf_f1:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')

# Random Forest için özel confusion matrix çizdirme
plot_confusion_matrix_custom(y_test, rf_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Random Forest Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Support Vector Machine modelini oluşturur ve eğitir
svm_model = SVC(random_state=500)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Support Vector Machine performans metriklerini hesaplar
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
svm_precision = precision_score(y_test, svm_pred, average='weighted', zero_division=1)
svm_recall = recall_score(y_test, svm_pred, average='weighted', zero_division=1)

# Sonuçları yazdırır
print('\nSupport Vector Machine Metrics:')
print(f'Accuracy: {svm_accuracy:.4f}')
print(f'F1 Score: {svm_f1:.4f}')
print(f'Precision: {svm_precision:.4f}')
print(f'Recall: {svm_recall:.4f}')

# Support Vector Machine için özel confusion matrix çizdirme
plot_confusion_matrix_custom(y_test, svm_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Support Vector Machine Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Gradient Boosting modelini oluşturur ve eğitir
gb_model = GradientBoostingClassifier(random_state=500)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Gradient Boosting performans metriklerini hesaplar
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred, average='weighted')
gb_precision = precision_score(y_test, gb_pred, average='weighted', zero_division=1)
gb_recall = recall_score(y_test, gb_pred, average='weighted', zero_division=1)

# Sonuçları yazdırır
print('\nGradient Boosting Metrics:')
print(f'Accuracy: {gb_accuracy:.4f}')
print(f'F1 Score: {gb_f1:.4f}')
print(f'Precision: {gb_precision:.4f}')
print(f'Recall: {gb_recall:.4f}')

# Gradient Boosting için özel confusion matrix çizdirme
plot_confusion_matrix_custom(y_test, gb_pred, ['NetBIOS', 'BENIGN', 'LDAP', 'MSSQL', 'Syn', 'UDP'], 'Gradient Boosting Confusion Matrix', cmap='Reds')

#------------------------------------------------

# Her bir model için ROC eğrisi çizimi
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

# Gradient Boosting için ROC
gb_prob = gb_model.predict_proba(X_test)
gb_prob_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 5, 6])
gb_fpr, gb_tpr, _ = roc_curve(gb_prob_bin.ravel(), gb_prob[:, 1])
gb_auc = auc(gb_fpr, gb_tpr)
plt.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {gb_auc:.2f})')

# Diğer ayarlamalar
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Toplu ROC Eğrisi')
plt.legend()
plt.show()


