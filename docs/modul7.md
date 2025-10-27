# **Modul 07:** Stratified K-Fold Cross Validation dengan Support Vector Machine (SVM)

## Tujuan:
1. Dapat membuat model machine learning menggunakan algoritma Support Vector Machine (SVM)
2. Memahami pengaruh kernel dan parameter pada performa SVM
3. Menerapkan stratified k-fold cross validation dan visualisasi hasil

## Langkah-langkah:

### 1. Baca dataset diabetes.csv
```python
import pandas as pd

filepath = 'diabetes.csv'
data = pd.read_csv(filepath)
```

### 2. Buat model machine learning SVM dengan fitur numerikal dan kategorikal
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
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

# Model SVM dengan kernel RBF (Radial Basis Function)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])
```

**Catatan:** 
- **kernel**: Fungsi kernel yang digunakan ('linear', 'poly', 'rbf', 'sigmoid')
- **C**: Parameter regularisasi (nilai lebih besar = lebih strict terhadap misklasifikasi)
- **gamma**: Koefisien kernel untuk 'rbf', 'poly', dan 'sigmoid'

### 3. Gunakan fungsi StratifiedKFold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

### 4. Siapkan variabel untuk menampung data performansi
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import time

confusion_matrices = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
training_times = []
```

### 5. Lakukan pelatihan dan prediksi untuk setiap fold
```python
for train_index, test_index in skf.split(X, y):
    # Membagi dataset ke dalam k-fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Latih model dan ukur waktu training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    training_times.append(training_time)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(recall)
    f1_scores.append(f1)
    
    # Simpan confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
```

### 6. Cetak total (averaged) performansi model
```python
print("="*50)
print("HASIL EVALUASI MODEL SVM")
print("="*50)
print(f"Akurasi rata-rata: {sum(accuracies)/len(accuracies):.4f} ± {pd.Series(accuracies).std():.4f}")
print(f"Precision rata-rata: {sum(precisions)/len(precisions):.4f} ± {pd.Series(precisions).std():.4f}")
print(f"Recall rata-rata: {sum(recalls)/len(recalls):.4f} ± {pd.Series(recalls).std():.4f}")
print(f"F1-Score rata-rata: {sum(f1_scores)/len(f1_scores):.4f} ± {pd.Series(f1_scores).std():.4f}")
print(f"Waktu training rata-rata: {sum(training_times)/len(training_times):.4f} detik")
print("="*50)
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax)
    ax.set_title(f'Fold {i+1} - Acc: {accuracies[i]:.3f}\nF1: {f1_scores[i]:.3f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrix untuk Setiap Fold (SVM - RBF Kernel)', 
             fontsize=16, y=1.00)
plt.tight_layout()
plt.show()
```

## Tugas:

### 1. Perbandingan Kernel SVM
Bandingkan performa SVM dengan kernel yang berbeda menggunakan 10-fold cross validation:

- Linear kernel: `SVC(kernel='linear', C=1.0)`

- Polynomial kernel: `SVC(kernel='poly', degree=3, C=1.0)`

- RBF kernel: `SVC(kernel='rbf', C=1.0, gamma='scale')`

- Sigmoid kernel: `SVC(kernel='sigmoid', C=1.0, gamma='scale')`

Untuk setiap kernel, tampilkan:

- Tabel perbandingan akurasi, precision, recall, dan F1-score

- Grafik bar chart untuk membandingkan metrik-metrik tersebut

- Waktu training rata-rata

- Confusion matrix untuk kernel dengan performa terbaik

Analisis: Kernel mana yang paling cocok untuk dataset diabetes? Jelaskan mengapa!

**Petunjuk:**
```python
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_results = {}

for kernel in kernels:
    print(f"\nEvaluasi kernel: {kernel}")
    if kernel == 'poly':
        model_svm = SVC(kernel=kernel, degree=3, C=1.0, random_state=42)
    else:
        model_svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_svm)
    ])
    # Lakukan cross validation
    # ...
```

### 2. Optimasi Parameter C dan Gamma untuk RBF Kernel
Parameter C dan gamma sangat mempengaruhi performa SVM dengan RBF kernel. Lakukan grid search untuk menemukan kombinasi terbaik:

- C values: [0.1, 1, 10, 100]

- gamma values: ['scale', 'auto', 0.001, 0.01, 0.1, 1]

Gunakan 5-fold cross validation untuk setiap kombinasi parameter. Tampilkan:

- Heatmap yang menunjukkan akurasi untuk setiap kombinasi C dan gamma

- Parameter optimal yang memberikan akurasi tertinggi

- Confusion matrix untuk model dengan parameter optimal

- Analisis: Apa pengaruh parameter C dan gamma terhadap performa model?

**Petunjuk:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print(f"Parameter terbaik: {grid_search.best_params_}")
print(f"Akurasi terbaik: {grid_search.best_score_:.4f}")
```

### 3. Perbandingan Kompleksitas Waktu (Bonus)
Bandingkan waktu training SVM (RBF kernel) dengan berbagai ukuran dataset: 500, 1000, 2000, 5000, 10000 data.
```python
data_sizes = [500, 1000, 2000, 5000, 10000]

for size in data_sizes:
    sampled_data = data.sample(n=min(size, len(data)), random_state=42)
    # Ukur waktu training
    # ...
```
Buat grafik yang menunjukkan hubungan antara jumlah data dengan waktu training. Apa kesimpulan Anda tentang scalability SVM?
