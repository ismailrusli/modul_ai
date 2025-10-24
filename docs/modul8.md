# Modul 08: Stratified K-Fold Cross Validation dengan Neural Network (MLP)

## Tujuan:
1. Dapat membuat model machine learning menggunakan algoritma Neural Network (Multi-Layer Perceptron)
2. Memahami arsitektur neural network dan parameter-parameternya
3. Menerapkan stratified k-fold cross validation dan analisis performa model

## Langkah-langkah:

### 1. Baca dataset diabetes.csv
```python
import pandas as pd
import numpy as np

filepath = 'diabetes.csv'
data = pd.read_csv(filepath)
```

### 2. Buat model machine learning Neural Network dengan fitur numerikal dan kategorikal
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
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

# Model Neural Network dengan 2 hidden layers
# Hidden layer 1: 64 neurons, Hidden layer 2: 32 neurons
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ))
])
```

**Catatan Parameter:**
- **hidden_layer_sizes**: Tuple yang mendefinisikan jumlah neurons di setiap hidden layer
- **activation**: Fungsi aktivasi ('relu', 'tanh', 'logistic')
- **solver**: Optimizer ('adam', 'sgd', 'lbfgs')
- **alpha**: Parameter regularisasi L2
- **learning_rate**: Strategi learning rate ('constant', 'adaptive', 'invscaling')
- **max_iter**: Jumlah maksimum epoch
- **early_stopping**: Menghentikan training jika tidak ada perbaikan

### 3. Gunakan fungsi StratifiedKFold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

### 4. Siapkan variabel untuk menampung data performansi
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import time
import warnings
warnings.filterwarnings('ignore')  # Untuk menghindari warning konvergensi

confusion_matrices = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
training_times = []
n_iterations = []  # Menyimpan jumlah iterasi hingga konvergen
```

### 5. Lakukan pelatihan dan prediksi untuk setiap fold
```python
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"Training Fold {fold}...", end=' ')
    
    # Membagi dataset ke dalam k-fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Latih model dan ukur waktu training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    training_times.append(training_time)
    
    # Simpan jumlah iterasi
    n_iter = model.named_steps['classifier'].n_iter_
    n_iterations.append(n_iter)
    
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
    
    print(f"Acc: {acc:.4f}, Time: {training_time:.2f}s, Iterations: {n_iter}")
```

### 6. Cetak total (averaged) performansi model
```python
print("\n" + "="*60)
print("HASIL EVALUASI MODEL NEURAL NETWORK (MLP)")
print("="*60)
print(f"Akurasi rata-rata: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision rata-rata: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall rata-rata: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1-Score rata-rata: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Waktu training rata-rata: {np.mean(training_times):.4f} detik")
print(f"Iterasi rata-rata hingga konvergen: {np.mean(n_iterations):.1f}")
print("="*60)
```

### 7. Visualisasi performa per fold
```python
import matplotlib.pyplot as plt

# Grafik metrik per fold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Metrik evaluasi per fold
ax1 = axes[0]
folds = range(1, len(accuracies) + 1)
ax1.plot(folds, accuracies, marker='o', label='Accuracy', linewidth=2)
ax1.plot(folds, precisions, marker='s', label='Precision', linewidth=2)
ax1.plot(folds, recalls, marker='^', label='Recall', linewidth=2)
ax1.plot(folds, f1_scores, marker='d', label='F1-Score', linewidth=2)
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Metrik Evaluasi per Fold', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(folds)

# Plot 2: Waktu training per fold
ax2 = axes[1]
ax2.bar(folds, training_times, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Fold', fontsize=12)
ax2.set_ylabel('Waktu Training (detik)', fontsize=12)
ax2.set_title('Waktu Training per Fold', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(folds)

plt.tight_layout()
plt.show()
```

### 8. Tentukan area display untuk confusion matrix
```python
import seaborn as sns

# Untuk 10-fold, gunakan 2 baris dan 5 kolom
n_rows, n_cols = 2, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))

# Buat agar index adalah list 1 dimensi
axes = axes.flatten()
```

### 9. Gambar masing-masing confusion matrix
```python
for i, cm in enumerate(confusion_matrices):
    ax = axes[i]
    
    # Gambar heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax)
    ax.set_title(f'Fold {i+1}\nAcc: {accuracies[i]:.3f} | F1: {f1_scores[i]:.3f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrix untuk Setiap Fold (Neural Network)', 
             fontsize=16, y=1.00, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Tugas:

### 1. Eksplorasi Arsitektur Neural Network
Bandingkan performa neural network dengan arsitektur yang berbeda menggunakan 10-fold cross validation:
- Shallow network: `hidden_layer_sizes=(32,)`
- Medium network: `hidden_layer_sizes=(64, 32)`
- Deep network: `hidden_layer_sizes=(128, 64, 32)`
- Very deep network: `hidden_layer_sizes=(128, 64, 32, 16)`

Untuk setiap arsitektur, tampilkan:
- Tabel perbandingan akurasi, precision, recall, F1-score, dan waktu training
- Grafik bar chart untuk membandingkan metrik
- Grafik jumlah iterasi hingga konvergen untuk setiap arsitektur
- Confusion matrix untuk arsitektur terbaik

Analisis: 
- Arsitektur mana yang memberikan performa terbaik?
- Apakah arsitektur yang lebih dalam selalu lebih baik? Jelaskan!
- Bagaimana trade-off antara kompleksitas model dan performa?

### 2. Pengaruh Fungsi Aktivasi dan Optimizer
Lakukan eksperimen dengan kombinasi fungsi aktivasi dan optimizer yang berbeda:

**Fungsi Aktivasi:**
- ReLU: `activation='relu'`
- Tanh: `activation='tanh'`
- Logistic (Sigmoid): `activation='logistic'`

**Optimizer:**
- Adam: `solver='adam'`
- SGD: `solver='sgd'`
- L-BFGS: `solver='lbfgs'`

Gunakan arsitektur `hidden_layer_sizes=(64, 32)` dan 5-fold cross validation untuk setiap kombinasi.

Tampilkan:
- Heatmap yang menunjukkan akurasi untuk setiap kombinasi aktivasi dan optimizer
- Tabel waktu training untuk setiap kombinasi
- Kombinasi terbaik beserta confusion matrix-nya
- Analisis: Mengapa kombinasi tertentu bekerja lebih baik?

**Petunjuk:**
```python
activations = ['relu', 'tanh', 'logistic']
solvers = ['adam', 'sgd', 'lbfgs']
results_matrix = np.zeros((len(activations), len(solvers)))

for i, activation in enumerate(activations):
    for j, solver in enumerate(solvers):
        model_mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation=activation,
            solver=solver,
            max_iter=500,
            random_state=42
        )
        # Lakukan cross validation dan simpan hasil
        # ...
```

### 3. Learning Curve dan Regularisasi (Bonus)
Analisis pengaruh regularisasi (parameter alpha) terhadap performa model:
- Alpha values: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

Untuk setiap nilai alpha:
1. Latih model dengan 5-fold cross validation
2. Hitung train score dan validation score
3. Plot learning curve yang menunjukkan train vs validation accuracy

Tampilkan:
- Grafik learning curve untuk setiap nilai alpha
- Grafik yang menunjukkan hubungan antara alpha dengan train/validation accuracy
- Identifikasi nilai alpha optimal yang mencegah overfitting
- Analisis: Bagaimana regularisasi mempengaruhi generalisasi model?

**Petunjuk:**
```python
from sklearn.model_selection import learning_curve

alpha_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

for alpha in alpha_values:
    model_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=alpha,
        max_iter=500,
        random_state=42
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_mlp)
    ])
    
    # Gunakan learning_curve untuk mendapatkan train dan validation scores
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    # Plot hasilnya
    # ...
```

## Catatan Penting:
1. Neural network memerlukan data yang sudah dinormalisasi (StandardScaler sangat penting)
2. Early stopping membantu mencegah overfitting
3. Learning rate adaptive membantu konvergensi lebih baik
4. Jumlah neurons dan layers mempengaruhi kapasitas model untuk belajar pola yang kompleks
5. Regularisasi (alpha) membantu mencegah overfitting pada model yang kompleks
