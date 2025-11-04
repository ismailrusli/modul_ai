# **Modul 09:** Klasifikasi Gambar Dataset MNIST

## Tujuan

1. Memahami deep learning 
1. Mempelajari dataset gambar tulisan tangan MNIST
1. Menggunakan [API Keras](https://keras.io/) untuk memuat dataset MNIST dan mempersiapkannya untuk training
1. Membuat jaringan saraf tiruan sederhana untuk melakukan klasifikasi gambar
1. Melatih jaringan saraf tiruan menggunakan dataset MNIST yang telah dipersiapkan
1. Mengamati performa jaringan sarah tiruan yang telah dilatih

## Persiapan
Sebelum memulai modul ini, beberapa modul python yang harus di-install adalah tensorflow dan matplotlib. Install dengan menjalankan perintah berikut di terminal.
```
pip install tensorflow matplotlib
```

## Load dataset MNIST

Dataset MNIST adalah koleksi 70.000 gambar tulisan tangan angka 0 sampai 9 dalam format grayscale. Biasanya, klasifikasi gambar dengan MNIST dianggap semacam "Hello World" untuk belajar deep learning.

Ada banyak [framework deep learning](https://developer.nvidia.com/deep-learning-frameworks) yang dapat digunakan. Kita akan menggunakan [Tensorflow 2](https://www.tensorflow.org/tutorials/quickstart/beginner) dengan [API Keras](https://keras.io/). Untuk load MNIST.

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

## Data MNIST
Keras membagi MNIST menjadi 60.000 data training dan 10.000 data testing. Setiap gambar MNIST berukuran 28 x 28 piksel.

```python
print(X_train.shape)
print(X_test.shape)
```

Sebuah gambar tulisan tangan dalam MNIST berisi 28 x 28 = 784 piksel yang masing-masing bernilai 0-255. Jika sebuah piksel nilainya 0, itu adalah piksel hitam. Sebaliknya, jika nilai piksel 255, itu adalah piksel putih.

```python
print(X_train.dtype)
print(X_train.min())
print(X_train.max())
print(X_train[0])
```
Jika ingin melihat satu gambar dari MNIST, gunakan perintah berikut.

```python
import matplotlib.pyplot as plt

image = X_train[0]
plt.imshow(image, cmap='gray')
plt.show()
```

Angka apa yang muncul? Angka 5 atau angka 3? Jawabannya ada di data training berikut.

```python
print(y_train[0])
```

## Mempersiapkan Data untuk Pelatihan
Sebelum digunakan, data perlu kita ubah dulu agar mudah untuk proses training. Tiga hal yang akan kita lakukan terhadap data MNIST adalah: 

1. Meratakan (flatten) data gambar
2. Menormalisasi data gambar
3. Mengkategorikan label

### Meratakan Data Gambar
Setiap gambar dalam dataset MNIST berukuran 28x28. Artinya, untuk setiap gambar terdapat 784 piksel. Untuk memudahkan, kita ubah dari 2 dimensi (28x28) menjadi 1 dimensi (1x784).

```python
X_train = X_train.reshape(60000, 784) # 60000 gambar masing-masing berisi 784 data
X_test = X_test.reshape(10000, 784)

print(X_train.shape)
print(X_train[0])
```

### Menormalisasi Data Gambar
Setiap piksel bernilai antara 0-255. Dalam perhitungannya, lebih baik menggunakan nilai rentang antara 0-1. Dengan rentang yang lebih kecil, perhitungan tidak akan menjadi sangat besar dan melebihi kapasitas suatu tipe data. Selain itu, perhitungan internal dalam deep learning juga lebih stabil karena berurusan dengan nilai yang relatif lebih kecil (misalnya dalam perhitungan gradien). Proses mengubah nilai rentang 0-255 ini disebut dengan [normalisasi](https://developers.google.com/machine-learning/glossary#normalization).

```python
X_train = X_train / 255
X_valid = X_valid / 255 

print(X_train.dtype)
print(X_train.min())
print(X_train.max())
```

### Mengkategorikan Label
Terdapat 10 label untuk MNIST, yaitu angka 0-9. Ketika satu data berlabel 1, misalnya, angka 1 ini tidak berarti jaraknya 2 dari label 3. Label tidak memiliki arti numerik. Label ini adalah kategori. Untuk itu, sebaiknya label ini tidak direpresentasikan dalam bilangan bulat 0-9. Untuk itu, kita melakukan encoding terhadap label ini. Sebagai contoh, jika kita punya 3 label, yaitu Merah, Biru, dan Hijau, kita dapat menggunakan encoding kategorikal seperti berikut.

|Warna Aktual| Merah? | Biru? | Hijau?|
|------------|---------|----------|----------|
|Merah|1|0|0|
|Hijau|0|0|1|
|Biru|0|1|0|
|Hijau|0|0|1|

Jadi, dalam encoding ini, Merah = [1,0,0], Biru = [0,1,0], dan Hijau = [0,0,1].
Keras menyediakan fungsi untuk [encoding kategorikal nilai](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical).

```python
import tensorflow.keras as keras
num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_test = keras.utils.to_categorical(y_test, num_categories)

print(y_train[0:9])
```

## Membuat Model
Setelah data MNIST disiapkan, sekarang kita akan membuat model deep learning. Jaringan saraf tiruan kita akan terdiri dari 3 layer, yaitu layer input, hidden layer, dan layer output. Kita akan menggunakan kelas [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) karena model kita akan dilalui data secara berurutan:

```python
from tensorflow.keras.models import Sequential

model = Sequential()
```

### Membuat Layer Input
Layer pertama kita adalah layer input. Layer ini akan terhubung dengan semua neuron di layer berikutnya. Untuk itu, kita akan menggunakan kelas [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) Keras.

```python
from tensorflow.keras.layers import Dense

model.add(Dense(units=512, activation='relu', input_shape=(784,)))
```

### Membuat Hidden Layer
Setelah input layer, kita buat layer hidden.

```python
model.add(Dense(units = 512, activation='relu'))
```

### Membuat Layer Output
Terakhir, kita buat layer output.

```python
model.add(Dense(units = 10, activation='softmax'))
```

### Merangkum Model

Keras menyediakan merangkum model kita menggunakan fungsi [summary](https://www.tensorflow.org/api_docs/python/tf/summary).

```python
model.summary()
```

### Mengkompilasi Model
Langkah selanjutnya adalah dengan melakukan [kompilasi](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile)terhadap jaringan saraf tiruan kita. Dalam proses ini, kita menentukan [fungsi loss](https://developers.google.com/machine-learning/glossary#loss) yang akan digunakan untuk menentukan kinerja model berdasarkan akurasinya. 

```python
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
```

## Melatih Model
Untuk melatih model, kita gunakan fungsi [fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fita). 

```python
history = model.fit(
    X_train, y_train, epochs=5, verbose=1, validation_data=(X_test, y_test)
)
```

Berapakah akurasi model yang sudah dilatih?

## Tugas
1. Lakukan prediksi untuk satu gambar dalam `X_test`, misalnya `X_test[0]`, lalu bandingkan hasilnya dengan `y_test[0]`. Tampilkan gambar yang diprediksi menggunakan matplotlib.
2. Cari di Internet (atau tanya AI) cara untuk menyimpan/meload model ke file sehingga dapat dapat digunakan berulang-ulang tanpa harus training terlebih dahulu.
