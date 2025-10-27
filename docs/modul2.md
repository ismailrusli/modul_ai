# **Modul 02:** Membaca Data

## Tujuan

1. Membaca data dari DataFrame
2. Menampilkan data sesuai kebutuhan

## Langkah-langkah

### 1. Persiapan Data

Pastikan data yang akan dibaca telah berada di Google Colab. Misalkan kita akan menggunakan sample data yang sudah ada di Google Colab, yaitu `california_housing_test.csv`.

### 2. Import Library dan Set Filepath

```python
import pandas as pd
filepath = 'sample_data/california_housing_test.csv'
```

### 3. Membaca Dataset

Gunakan `pd.read_csv()` untuk membaca dataset dari file CSV.

```python
df = pd.read_csv('path_to_your_dataset.csv')
```

### 4. Menampilkan Informasi Dasar dari Tabel

Gunakan fungsi-fungsi berikut untuk melihat informasi dasar:
- `df.head()` untuk melihat beberapa baris pertama dari tabel
- `df.info()` untuk menampilkan informasi tentang jumlah kolom, tipe data, dan jumlah non-null
- `df.describe()` untuk mendapatkan statistik deskriptif dasar dari kolom numerik

```python
print(df.head())
print(df.info())
print(df.describe())
```

### 5. Mengecek Missing Values

Gunakan `df.isnull().sum()` untuk mengetahui jumlah missing values di setiap kolom.

```python
print(df.isnull().sum())
```

### 6. Membuang Kolom Menggunakan drop()

Gunakan fungsi `drop()` untuk menghapus kolom yang tidak diperlukan.

```python
df_cleaned = df.drop(['nama_kolom_1', 'nama_kolom_2'], axis=1)
```

### 7. Verifikasi Hasil

Gunakan `df_cleaned.info()` untuk memverifikasi bahwa kolom yang tidak diperlukan telah dihapus.

```python
print(df_cleaned.info())
```

### 8. Mengambil Kolom Tertentu

Gunakan sintaks `df['nama_kolom']` untuk mengambil satu kolom dari dataset.

```python
specific_column = df['nama_kolom']
specific_column.head()
```

### 9. Analisis Kolom Tersebut

Lakukan analisis sederhana seperti menghitung nilai rata-rata, distribusi nilai, atau menemukan nilai unik.

```python
print(specific_column.mean())  # contoh untuk data numerik
print(specific_column.value_counts())  # contoh untuk data kategorikal
```

## Tugas

1. Cek tipe data dari variabel df menggunakan fungsi `type`.
2. Cari dokumentasi dari tipe data DataFrame di website pandas dan lakukan eksplorasi fungsi-fungsi yang bisa dipanggil untuk DataFrame.
3. Bagaimana cara menampilkan semua atribut kolom?
4. Bagaimana cara menampilkan satu kolom?
5. Bagaimana cara menampilkan dua kolom?
6. Bagaimana cara menampilkan baris-baris dengan batasan tertentu, misalnya hanya baris-baris yang `total_rooms > 500` atau `total_bedrooms < 200`?
7. Bagaimana cara menghitung rata-rata suatu kolom?
