# **Modul 10:** Klasifikasi Gambar Dataset MNIST dengan Convolutional Neural Network (CNN)

## Tujuan

1. Memahami arsitektur Convolutional Neural Network (CNN)
2. Mempelajari perbedaan CNN dengan MLP untuk klasifikasi gambar
3. Menggunakan [API Keras](https://keras.io/) untuk memuat dataset MNIST
4. Membuat CNN untuk melakukan klasifikasi gambar
5. Melatih CNN menggunakan dataset MNIST
6. Membandingkan performa CNN dengan MLP

## Persiapan
Sebelum memulai modul ini, beberapa modul python yang harus di-install adalah tensorflow dan matplotlib. Install dengan menjalankan perintah berikut di terminal.
```
pip install tensorflow matplotlib
```

## Load dataset MNIST

Dataset MNIST adalah koleksi 70.000 gambar tulisan tangan angka 0 sampai 9 dalam format grayscale. Biasanya, klasifikasi gambar dengan MNIST dianggap semacam "Hello World" untuk belajar deep learning.

Kita akan menggunakan [Tensorflow 2](https://www.tensorflow.org/tutorials/quickstart/beginner) dengan [API Keras](https://keras.io/) untuk load MNIST.

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

## Mempersiapkan Data untuk Pelatihan CNN

Berbeda dengan MLP yang memerlukan data dalam bentuk 1 dimensi, CNN bekerja dengan data gambar dalam bentuk aslinya (2D untuk grayscale, 3D untuk RGB). Tiga hal yang akan kita lakukan terhadap data MNIST adalah:

1. Reshape data gambar untuk CNN
2. Menormalisasi data gambar
3. Mengkategorikan label

### Reshape Data Gambar untuk CNN

CNN memerlukan input dalam format (jumlah_data, tinggi, lebar, channel). Untuk gambar grayscale seperti MNIST, channel = 1. Kita akan reshape data dari (28, 28) menjadi (28, 28, 1).

```python
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

print(X_train.shape)
print(X_test.shape)
```

### Menormalisasi Data Gambar

Setiap piksel bernilai antara 0-255. Dalam perhitungannya, lebih baik menggunakan nilai rentang antara 0-1. Dengan rentang yang lebih kecil, perhitungan tidak akan menjadi sangat besar dan melebihi kapasitas suatu tipe data. Selain itu, perhitungan internal dalam deep learning juga lebih stabil karena berurusan dengan nilai yang relatif lebih kecil (misalnya dalam perhitungan gradien). Proses mengubah nilai rentang 0-255 ini disebut dengan [normalisasi](https://developers.google.com/machine-learning/glossary#normalization).

```python
X_train = X_train / 255.0
X_test = X_test / 255.0

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

## Membuat Model CNN

Setelah data MNIST disiapkan, sekarang kita akan membuat model CNN. CNN terdiri dari beberapa layer konvolusi dan pooling untuk ekstraksi fitur, diikuti dengan layer fully connected untuk klasifikasi. Kita akan menggunakan kelas [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential):

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
```

### Membuat Layer Konvolusi Pertama

Layer konvolusi pertama akan mengekstrak fitur-fitur sederhana dari gambar. Kita gunakan 32 filter dengan ukuran 3x3. Activation function 'relu' digunakan untuk menambah non-linearity.

```python
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
```

### Membuat Layer Pooling Pertama

Layer pooling digunakan untuk mengurangi dimensi spatial dari output konvolusi, sehingga mengurangi jumlah parameter dan komputasi dalam jaringan.

```python
model.add(MaxPooling2D(pool_size=(2, 2)))
```

### Membuat Layer Konvolusi Kedua

Layer konvolusi kedua akan mengekstrak fitur-fitur yang lebih kompleks. Kita gunakan 64 filter dengan ukuran 3x3.

```python
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
```

### Membuat Layer Pooling Kedua

```python
model.add(MaxPooling2D(pool_size=(2, 2)))
```

### Membuat Layer Flatten

Layer flatten mengubah output 2D dari layer konvolusi menjadi 1D array yang dapat diproses oleh layer Dense.

```python
model.add(Flatten())
```

### Membuat Layer Fully Connected

Layer Dense pertama adalah layer fully connected dengan 128 neuron.

```python
model.add(Dense(128, activation='relu'))
```

### Membuat Layer Dropout

Dropout adalah teknik regularisasi yang membantu mencegah overfitting dengan secara acak "mematikan" beberapa neuron selama training.

```python
model.add(Dropout(0.5))
```

### Membuat Layer Output

Layer output memiliki 10 neuron (untuk 10 kelas digit 0-9) dengan activation function 'softmax' untuk menghasilkan probabilitas untuk setiap kelas.

```python
model.add(Dense(10, activation='softmax'))
```

### Merangkum Model

Keras menyediakan fungsi untuk merangkum model kita menggunakan fungsi [summary](https://www.tensorflow.org/api_docs/python/tf/summary).

```python
model.summary()
```

Perhatikan jumlah parameter dalam model CNN ini. Bandingkan dengan model MLP yang memiliki jutaan parameter!

### Mengkompilasi Model

Langkah selanjutnya adalah dengan melakukan [kompilasi](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile) terhadap jaringan saraf tiruan kita. Dalam proses ini, kita menentukan optimizer, [fungsi loss](https://developers.google.com/machine-learning/glossary#loss) yang akan digunakan untuk menentukan kinerja model berdasarkan akurasinya.

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## Melatih Model

Untuk melatih model, kita gunakan fungsi [fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

```python
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_data=(X_test, y_test)
)
```

Berapakah akurasi model yang sudah dilatih? Anda akan melihat bahwa CNN biasanya mencapai akurasi yang lebih tinggi dibandingkan MLP pada dataset MNIST.

## Visualisasi Training History

Kita dapat memvisualisasikan bagaimana akurasi dan loss berubah selama proses training.

```python
import matplotlib.pyplot as plt

# Plot akurasi
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## Keunggulan CNN dibandingkan MLP

1. **Parameter Sharing**: CNN menggunakan filter yang sama untuk seluruh gambar, sehingga jumlah parameter jauh lebih sedikit
2. **Spatial Hierarchy**: CNN dapat menangkap pola-pola lokal dan hierarki fitur dari sederhana ke kompleks
3. **Translation Invariance**: CNN dapat mengenali pola yang sama di berbagai posisi dalam gambar
4. **Akurasi Lebih Tinggi**: CNN biasanya memberikan akurasi yang lebih baik untuk tugas-tugas computer vision

## Tugas

1. Lakukan prediksi untuk satu gambar dalam `X_test`, misalnya `X_test[0]`, lalu bandingkan hasilnya dengan `y_test[0]`. Tampilkan gambar yang diprediksi menggunakan matplotlib dan tampilkan probabilitas untuk setiap kelas.
2. Cari di Internet (atau tanya AI) cara untuk menyimpan/meload model ke file sehingga dapat digunakan berulang-ulang tanpa harus training terlebih dahulu.
3. Eksperimen dengan arsitektur CNN yang berbeda:

    * Tambahkan lebih banyak layer konvolusi
    * Ubah jumlah filter
    * Ubah ukuran kernel
    * Bandingkan akurasi yang dihasilkan

4. Bandingkan waktu training dan akurasi antara model MLP (dari tutorial sebelumnya) dengan model CNN ini. Buat tabel perbandingan hasilnya.

<!-- 4. Visualisasikan filter/kernel yang dipelajari oleh layer konvolusi pertama untuk melihat fitur apa yang dideteksi oleh CNN. -->
