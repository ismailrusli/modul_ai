# **Modul 03:** Membuat Model Machine Learning Menggunakan Random Forest

## Tujuan

1. Menyiapkan data untuk membuat model machine learning
2. Membuat model machine learning dan menguji akurasinya

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

### 3. Membuat Dataset dengan Fitur Numerik

Target kita adalah membuat model machine learning yang dapat memprediksi seseorang menderita diabetes atau tidak dari beberapa data. Kita akan mulai menggunakan data numerik terlebih dahulu, yaitu data `'age'`, `'bmi'`, `'HbA1c_level'`, dan `'blood_glucose_level'`.

```python
kolom = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
dataset = data[kolom]
dataset.info()
```

**Output:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 5 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   age                   100000 non-null  float64
 1   bmi                   100000 non-null  float64
 2   HbA1c_level           100000 non-null  float64
 3   blood_glucose_level   100000 non-null  int64  
 4   diabetes              100000 non-null  int64  
dtypes: float64(3), int64(2)
memory usage: 3.8 MB
```

### 4. Memisahkan Fitur dan Kelas

Kita memisahkan dataset menjadi 2 bagian, yaitu data fitur dan data kelas. Fitur adalah data yang digunakan untuk memprediksi. Kelas adalah data yang diprediksi.

```python
X = dataset.drop('diabetes', axis=1)
y = dataset['diabetes']
```

### 5. Membagi Data Training dan Testing

Masing-masing X dan y memiliki 100.000 baris. Untuk keperluan machine learning, kita akan menggunakan 80% data untuk data latih (training) dan sisanya (20%) untuk data uji (testing).

```python
X_train = X[:80000]
y_train = y[:80000]
X_test = X[80000:]
y_test = y[80000:]
```

### 6. Import Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
```

### 7. Membuat dan Melatih Model

Gunakan fungsi `RandomForestClassifier` untuk membuat model dan latih menggunakan fungsi `fit`.

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 8. Menguji Model dengan Satu Data

Kita sekarang memiliki model machine learning untuk memprediksi seseorang menderita diabetes atau tidak. Mari kita uji menggunakan 1 data uji dari `X_test`.

```python
data_uji1 = X_test.iloc[0]
print(data_uji1)
```

**Output:**
```
age                      25.00
bmi                      26.38
HbA1c_level               6.50
blood_glucose_level     100.00
Name: 80000, dtype: float64
```

### 9. Melihat Label Sebenarnya

```python
hasil_uji1 = y_test.iloc[0]
print(hasil_uji1)
```

**Output:**
```
0
```

### 10. Prediksi dengan Model

Jika kita cek menggunakan model, maka hasilnya adalah benar. Model kita memprediksi bahwa seseorang dengan data tersebut tidak memiliki penyakit diabetes.

```python
pred = model.predict(data_uji1.to_frame().T)
print(pred)
```

**Output:**
```
[0]
```

### 11. Import Accuracy Score

```python
from sklearn.metrics import accuracy_score
```

### 12. Menghitung Akurasi untuk Seluruh Data Testing

Selanjutnya kita lakukan prediksi untuk seluruh data di `X_test` yang hasilnya kita simpan di `y_pred`. Untuk menghitung akurasi, kita bandingkan `y_pred` dengan `y_test`.

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model = {accuracy:.2f}")
```

**Output:**
```
Akurasi model = 0.97
```

## Tugas

### 1. Aplikasi Prediksi Diabetes

Buatlah satu aplikasi untuk memprediksi seseorang kemungkinan menderita diabetes atau tidak berdasarkan 4 data, yaitu umur, bmi, tingkat HbA1c, dan tingkat glukosa dalam darah. Minta user memasukkan satu-satu data dan setelah selesai, aplikasi memberikan prediksi antara kemungkinan besar user menderita diabetes atau tidak.

**Petunjuk:** Data harus dikumpulkan dan dikemas dalam tipe data DataFrame sebelum diinputkan ke fungsi predict. Berikut adalah contoh mengemas data dalam DataFrame:

```python
age = 50
bmi = 20
HbA1c_level = 0.5
blood_glucose_level = 2
df = pd.DataFrame({
    'age': [10],
    'bmi': [20],
    'HbA1c_level': [0.5],
    'blood_glucose_level': [2]
})
y_pred = model.predict(df)
if y_pred:
    print("Anda kemungkinan besar menderita diabetes")
    print("Segera hubungi dokter.")
else:
    print("Anda kecil kemungkinan menderita diabetes")
```

### 2. Menggunakan train_test_split

Untuk membagi data menjadi data latih (X_train, y_train) dan data uji (X_test, y_test), kita dapat menggunakan `train_test_split` dari modul `sklearn.model_selection`. Modifikasi kode sebelumnya menggunakan fungsi ini.

```python
from sklearn.model_selection import train_test_split

X = dataset.drop('diabetes', axis=1)
y = dataset['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
