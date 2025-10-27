# **Modul 05:** Stratified K-Fold Cross Validation

## Tujuan

1. Dapat membuat model machine learning menggunakan teknik stratified k-fold cross validation
2. Menampilkan confusion matrix menggunakan pustaka matplotlib

## Langkah-langkah

### 1. Membaca Dataset

```python
import pandas as pd
filepath = 'diabetes.csv'
data = pd.read_csv(filepath)
```

### 2. Membuat Model Machine Learning

Buat model machine learning dengan fitur numerikal dan kategorikal seperti di modul 04.

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history']

X = dataset[numerical_features + categorical_features]
y = dataset['diabetes']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='passthrough',
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])
```

### 3. Menggunakan StratifiedKFold

Gunakan fungsi `StratifiedKFold` yang sudah tersedia di modul scikit-learn.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10)
```

### 4. Menyiapkan Variabel untuk Performansi

Siapkan variabel untuk menampung data performansi dari model machine learning kita.

```python
confusion_matrices = []
accuracies = []
precisions = []
recalls = []
```

### 5. Melakukan Pelatihan dan Prediksi untuk Setiap Fold

Lakukan pelatihan dan prediksi untuk setiap fold lewat perulangan.

```python
for train_index, test_index in skf.split(X, y):
    # Tugas skf.split adalah membagi dataset (X,y) ke dalam k-fold dan menyimpannya
    # dalam variabel X_train, X_test, y_train, dan y_test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Latih model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(recall)
    
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
```

### 6. Mencetak Total (Averaged) Performansi

Cetak total (averaged) akurasi, precision, dan recall.

```python
print(f"Akurasi total adalah: {sum(accuracies)/len(accuracies):.2f}")
print(f"Recall total adalah: {sum(recalls)/len(recalls):.2f}")
print(f"Precision total adalah: {sum(precisions)/len(precisions):.2f}")
```

### 7. Menentukan Area Display untuk Confusion Matrix

Kita akan menampilkan confusion matrix menggunakan bantuan pustaka matplotlib dan seaborn. Untuk itu pertama-tama kita tentukan dulu area displaynya.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Karena kita memiliki 5 model (5-fold), kita akan tampilkan
# confusion matrix dalam 2 baris (n_rows) dan 3 kolom (n_cols)
# Lalu, area gambar kita sebesar
# n_rows x 4 inch (tinggi) dan n_cols x 5 inch (lebar)
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

# Buat agar index adalah list 1 dimensi (bukan list dari list)
axes = axes.flatten()

# Hapus area gambar indeks 5 (gambar ke-6) karena kita hanya
# akan menggambar 5 confusion matrix
fig.delaxes(axes[5])
```

### 8. Menggambar Confusion Matrix

Lalu kita gambar masing-masing confusion matrix menggunakan perulangan.

```python
for i, cm in enumerate(confusion_matrices):
    # axes berarti posisi gambar di dalam area gambar
    ax = axes[i]
    
    # Gambar heatmap dari data cm
    # fmt = format, d = decimal integer, cmap = warna, cbar = legend
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix Fold {i+1}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
```

### 9. Menampilkan Gambar

Terakhir, tampilkan gambar.

```python
plt.tight_layout()  # supaya setiap gambar tidak saling bersinggungan
plt.show()
```

## Tugas

### 1. Perbandingan K-Fold Cross Validation dan Tanpa Cross Validation

Buat model machine learning untuk memprediksi diabetes dengan menggunakan Random Forest. Gunakan 5, 10, dan 15 k-fold cross validation. Tampilkan confusion matrix-nya dan bandingkan rata-rata masing-masing model machine learning. Manakah yang akurasinya paling tinggi? Bandingkan juga dengan tanpa cross-validation (untuk pembagian 60:40, 70:30, 80:20, dan 90:10).

### 2. Pengaruh Jumlah Data terhadap Performansi

Gunakan jumlah data yang berbeda. Misalnya, untuk k=5, cari akurasi, precision, dan recall untuk masing-masing dataset dengan jumlah 400, 1000, 4000, 10000, 40000, dan 100000. Untuk mengambil data secara acak dari dataset sebanyak 400, misalnya, gunakan perintah:

```python
data.sample(n=400)  # data adalah dataframe
```
