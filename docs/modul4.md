# **Modul 04:** Membuat Model Machine Learning Menggunakan Random Forest dengan Data Numerikal dan Kategorikal

## Tujuan

1. Menyiapkan data untuk membuat model machine learning
2. Membuat model machine learning dengan data numerikal dan kategorikal dan menguji akurasinya

## Langkah-langkah

### 1. Membaca Dataset

```python
import pandas as pd
filepath = 'diabetes.csv'
data = pd.read_csv(filepath)
```

### 2. Melihat Informasi Dataset

Dalam dataset diabetes, terdapat 9 kolom seperti yang terlihat dari perintah berikut dan luarannya.

```python
data.info()
```

**Output:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 9 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   gender                100000 non-null  object 
 1   age                   100000 non-null  float64
 2   hypertension          100000 non-null  int64  
 3   heart_disease         100000 non-null  int64  
 4   smoking_history       100000 non-null  object 
 5   bmi                   100000 non-null  float64
 6   HbA1c_level           100000 non-null  float64
 7   blood_glucose_level   100000 non-null  int64  
 8   diabetes              100000 non-null  int64  
dtypes: float64(3), int64(4), object(2)
memory usage: 6.9+ MB
```

### 3. Menentukan Fitur Numerikal dan Kategorikal

Target kita adalah membuat model machine learning yang dapat memprediksi seseorang menderita diabetes dari data numerikal dan kategorikal. Pertama-tama kita buat list yang memisahkan fitur numerikal dan fitur kategorikal.

```python
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'hypertension', 'smoking_history']
```

### 4. Memisahkan Fitur dan Label

Fitur kita pisahkan ke variabel X dan kelas/label kita masukkan ke variabel y.

```python
X = dataset[numerical_features + categorical_features]
y = dataset['diabetes']
```

### 5. Membuat Transformer untuk Data Numerikal

Suatu data dapat memiliki rentang yang sangat berbeda-beda. Misalnya, age rentang angkanya antara 0-90 sementara HbA1c_level mungkin rentangnya antara 1-10. Dalam perhitungan, perbedaan ini akan berpengaruh terhadap hasil akhir. Untuk itu, ada baiknya rentang waktu ini relatif disamakan. Kita melakukan transformasi terhadap data numerikal dengan memanggil fungsi `StandardScaler` agar skala (range) tiap data relatif sama.

### 6. Membuat Transformer untuk Data Kategorikal

Untuk data kategorikal, kita tidak bisa menerapkan `StandardScaler`. Akan tetapi, karena mesin hanya bekerja dengan data numerikal, kita harus mengubah data kategorikal ke dalam data numerikal. Untuk itu digunakan fungsi `OneHotEncoder`. Fungsi ini mengubah data kategorikal ke dalam vektor yang nilainya 1 dan 0.

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
```

### 7. Membuat Preprocessor dengan ColumnTransformer

Di langkah sebelumnya, kita mempersiapkan fungsi `StandardScaler` dan `OneHotEncoder`. Langkah selanjutnya menyiapkan transformasi tersebut untuk diterapkan ke data yang kita miliki. Untuk itu, kita panggil fungsi `ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='passthrough',
)
```

### 8. Membuat Pipeline Model

Selanjutnya, kita siapkan model `RandomForestClassifier` yang akan digunakan untuk data numerikal dan kategorikal yang sudah kita siapkan bersama transformernya. Model ini akan dibuat pertama lewat preprocessor dan dilanjutkan oleh classifier.

```python
from sklearn.ensemble import RandomForestClassifier

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])
```

### 9. Membagi Data dan Melatih Model

Sekarang, kita siapkan data training dan data testing kita dari X dan y menggunakan fungsi `train_test_split`. Selanjutnya, kita latih menggunakan fungsi `fit`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
```

### 10. Menguji Akurasi Model

Model yang sudah kita latih, kita uji dengan data testing. Berapakah akurasi model ini?

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

## Tugas

1. **Model A**: Buat model untuk memprediksi seseorang memiliki penyakit jantung atau tidak (`'heart_disease'`) dari data `'age'`, `'bmi'`, `'gender'`, `'hypertension'`, dan `'smoking_history'`.

2. **Model B**: Buat model untuk memprediksi seseorang memiliki penyakit jantung atau tidak (`'heart_disease'`) dari data `'age'`, `'HbA1c_level'`, `'gender'`, `'hypertension'`, dan `'smoking_history'`.

3. **Model C**: Buat model untuk memprediksi seseorang memiliki penyakit jantung atau tidak (`'heart_disease'`) dari data `'age'`, `'blood_glucose_level'`, `'gender'`, `'hypertension'`, dan `'smoking_history'`.

4. Mana yang lebih akurat, model A, B, atau C?
