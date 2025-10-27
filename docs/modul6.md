# **Modul 06:** Stratified K-Fold Cross Validation dengan K-Nearest Neighbors (KNN)

## Tujuan:
1. Dapat membuat model machine learning menggunakan algoritma K-Nearest Neighbors (KNN)
2. Menerapkan teknik stratified k-fold cross validation pada model KNN
3. Menampilkan confusion matrix dan menganalisis performa model

## Langkah-langkah:

### 1. Baca dataset diabetes.csv
```python
import pandas as pd

filepath = 'diabetes.csv'
data = pd.read_csv(filepath)
```

### 2. Buat model machine learning KNN dengan fitur numerikal dan kategorikal
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history']

X = data[numerical_features + categorical_features]
y = data['diabetes']

# Preprocessing
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='passthrough',
)

# Model KNN dengan k=5 sebagai default
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])
```

**Catatan:** Algoritma KNN sangat sensitif terhadap skala fitur, sehingga StandardScaler sangat penting untuk fitur numerikal.

### 3. Gunakan fungsi StratifiedKFold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

### 4. Siapkan variabel untuk menampung data performansi
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

confusion_matrices = []
accuracies = []
precisions = []
recalls = []
```

### 5. Lakukan pelatihan dan prediksi untuk setiap fold
```python
for train_index, test_index in skf.split(X, y):
    # Membagi dataset ke dalam k-fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Latih model
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(recall)
    
    # Simpan confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
```

### 6. Cetak total (averaged) akurasi, precision, dan recall
```python
print(f"Akurasi rata-rata: {sum(accuracies)/len(accuracies):.4f}")
print(f"Precision rata-rata: {sum(precisions)/len(precisions):.4f}")
print(f"Recall rata-rata: {sum(recalls)/len(recalls):.4f}")
print(f"\nStandar Deviasi Akurasi: {pd.Series(accuracies).std():.4f}")
```

### 7. Tentukan area display untuk confusion matrix
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Untuk 10-fold, gunakan 2 baris dan 5 kolom
n_rows, n_cols = 2, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))

# Buat agar index adalah list 1 dimensi
axes = axes.flatten()
```

### 8. Gambar masing-masing confusion matrix
```python
for i, cm in enumerate(confusion_matrices):
    ax = axes[i]
    
    # Gambar heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix Fold {i+1}\nAcc: {accuracies[i]:.3f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()
```

## Tugas:

### 1. Optimasi Parameter K
Buat eksperimen dengan nilai k yang berbeda (k = 3, 5, 7, 9, 11, 15, 21) menggunakan 10-fold cross validation. 

- Tampilkan grafik perbandingan akurasi rata-rata untuk setiap nilai k

- Tentukan nilai k optimal untuk dataset diabetes

- Jelaskan mengapa nilai k tersebut memberikan performa terbaik

**Petunjuk:**
```python
k_values = [3, 5, 7, 9, 11, 15, 21]
k_results = []

for k in k_values:
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k))
    ])
    # Lakukan cross validation dan simpan hasilnya
    # ...
```

### 2. Perbandingan dengan Metode Distance
KNN dapat menggunakan berbagai metode perhitungan jarak. Bandingkan performa model dengan:

- Euclidean distance (default)

- Manhattan distance (metric='manhattan')

- Minkowski distance dengan p=3 (metric='minkowski', p=3)

Gunakan k=5 dan 10-fold cross validation. Tampilkan:

- Tabel perbandingan akurasi, precision, dan recall

- Confusion matrix untuk metode terbaik

- Analisis: metode distance mana yang paling cocok untuk dataset diabetes dan mengapa?

### 3. Analisis Pengaruh Jumlah Data (Bonus)
Uji performa model KNN (k=5) dengan jumlah data yang berbeda: 500, 1000, 2000, 5000, 10000, dan seluruh dataset.
```python
data_sizes = [500, 1000, 2000, 5000, 10000, len(data)]

for size in data_sizes:
    sampled_data = data.sample(n=size, random_state=42)
    # Latih model dan evaluasi
    # ...
```
Buat grafik yang menunjukkan hubungan antara jumlah data dengan akurasi model.
