# Modul 01: Menggunakan Google Colab

## Tujuan Praktikum

1. Peserta mampu membuat file Google Colab di Google Drive
2. Peserta mampu mengakses file di Google Colab

## Pendahuluan

Google Colab adalah layanan dari Google yang memungkinkan Anda menulis dan mengeksekusi kode Python langsung di browser Anda, dengan lingkungan yang dilengkapi dengan GPU dan TPU. Ini sangat bermanfaat untuk proyek-proyek yang melibatkan pemrosesan data besar atau pelatihan model machine learning.

## Membuat file Google Colab di Google Drive

1. Bukalah Google Drive.
2. Di Folder utama (My Drive), buatlah satu folder untuk tempat file-file Google Colab (klik New > New Folder). Beri nama folder baru (misalkan file_ai).
3. Masuk ke folder file_ai lalu klik New > More > Google Colaboratory. Akan muncul tab browser baru yang memperlihatkan ruang kerja (workspace) Google Colab. Ganti nama file Google Colab di sebelah kiri atas dengan nama yang deskriptif (misal modul01.ipynb). Perhatikan Gambar 1.
4. Anda telah siap untuk membuat program di Google Colab.
5. Untuk mengakses file di Google Colab, kita harus menguploadnya. Cara lain adalah dengan menempelkan Google Drive jika file yang kita akan gunakan berada di Google Drive.
6. Untuk melihat file yang bisa diakses di Google Colab, klik gambar folder di panel sebelah kiri (Gambar 2). Tunggu beberapa saat hingga muncul daftar folder dan file seperti terlihat di Gambar 3.
7. Untuk upload file gunakan ikon upload dan untuk pasang Google Drive gunakan ikon Google Drive. Perhatikan Gambar 3.
8. Untuk mengakses file dalam kode dan membacanya menggunakan library pandas, tuliskan kode berikut.

```python
file = 'sample_data/california_housing_test.csv'
import pandas as pd
data = pd.read_csv(file)
print(data)
```

9. Hasil dari kode dapat dilihat di Gambar 4.

## Latihan

1. Upload file format csv (comma separated values) dari folder lokal (laptop atau PC)
2. Baca file menggunakan library pandas dan tampilkan
3. Tempelkan Google Drive ke Google Colab
4. Baca file dari Google Drive menggunakan library pandas dan tampilkan