# Modul 11: Pengolahan Citra Digital

Dalam tutorial ini, digunakan gambar berikut. Klik kanan dan "Save Image As ..." cat.jpg.

![cat.jpg](https://raw.githubusercontent.com/ismailrusli/grafika_citra/refs/heads/main/cat.jpg)

## Bagian 1: Pengolahan Dasar Citra Digital

### Tujuan
1. Mengetahui ukuran piksel pada citra
2. Mengonversi citra RGB menjadi Grayscale
3. Mengambil channel R, G, B, dan menampilkannya sebagai citra grayscale
4. Mengetahui perbedaan citra RGB, Grayscale, dan Black & White

### Pendahuluan
Citra digital adalah representasi visual dalam bentuk matriks angka. Pada praktikum ini, Anda akan mempelajari beberapa operasi dasar untuk memahami sifat dasar citra digital, termasuk struktur RGB, konversi citra, dan analisis channel.

### Alat dan Bahan
1. **Software:** Python dengan library berikut:
   - OpenCV (cv2)
   - Matplotlib
   - NumPy
2. **Dataset:** File gambar format .jpg atau .png

### Langkah-Langkah Praktikum

#### Langkah 1: Mengetahui Ukuran Piksel pada Citra

**Kode Program:**
```python
import cv2

# Baca citra
img = cv2.imread('cat.jpg')

# Dapatkan ukuran citra
height, width, channels = img.shape
print(f'Ukuran citra: {width} x {height}')
print(f'Jumlah channel: {channels}')
```

**Hasil yang Diharapkan:**
- Dimensi citra dalam piksel (lebar x tinggi)
- Jumlah channel (3 untuk RGB)

#### Langkah 2: Konversi RGB ke Grayscale

**Kode Program:**
```python
import cv2

# Baca citra
img = cv2.imread('cat.jpg')

# Konversi ke Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tampilkan hasil
cv2.imshow("original", img)
cv2.imshow("result", gray)
cv2.waitKey()
```

**Hasil yang Diharapkan:**
- Citra grayscale dengan intensitas warna hitam dan putih

#### Langkah 3: Mengambil Channel R, G, dan B

**Kode Program:**
```python
import cv2
import matplotlib.pyplot as plt

# Baca citra
img = cv2.imread('cat.jpg')

# Ambil channel
B, G, R = cv2.split(img)

# Tampilkan masing-masing channel sebagai Grayscale
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(R, cmap='gray')
plt.title('Red Channel')

plt.subplot(1, 3, 2)
plt.imshow(G, cmap='gray')
plt.title('Green Channel')

plt.subplot(1, 3, 3)
plt.imshow(B, cmap='gray')
plt.title('Blue Channel')

plt.tight_layout()
plt.show()
```

**Hasil yang Diharapkan:**
- Tiga gambar terpisah, masing-masing menunjukkan intensitas warna merah, hijau, dan biru dalam bentuk grayscale

#### Langkah 4: Konversi Grayscale ke Black & White

**Kode Program:**
```python
import cv2
import matplotlib.pyplot as plt

# Baca citra
img = cv2.imread('cat.jpg')

# Konversi ke Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Konversi ke Black & White dengan threshold
_, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Tampilkan hasil
cv2.imshow("original", img)
cv2.imshow("result", bw)
cv2.waitKey()
```

**Hasil yang Diharapkan:**
- Citra biner hanya memiliki dua warna: hitam dan putih

#### Langkah 5: Perbandingan RGB, Grayscale, dan Black & White

**Kode Program:**
```python
import cv2
import matplotlib.pyplot as plt

# Baca citra
img = cv2.imread('cat.jpg')

# Konversi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Tampilkan hasil perbandingan
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('RGB Image')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 3, 3)
plt.imshow(bw, cmap='gray')
plt.title('Black & White Image')

plt.tight_layout()
plt.show()
```

**Hasil yang Diharapkan:**
- Tampilan citra dalam tiga format (RGB, Grayscale, Black & White) untuk membandingkan perbedaannya

### Tugas untuk Mahasiswa
1. Coba gunakan gambar lain dan amati perbedaan ukuran serta channel
2. Jelaskan bagaimana citra RGB berbeda dari Grayscale dan Black & White berdasarkan hasil praktikum

---

## Bagian 2: Operasi Augmentasi pada Citra

### Tujuan Praktikum
1. Memahami konsep augmentasi citra
2. Melakukan berbagai operasi augmentasi seperti rotasi, flipping, scaling, brightness adjustment, dan penambahan noise
3. Menampilkan hasil augmentasi citra

### Pendahuluan
Image Augmentation adalah teknik untuk meningkatkan jumlah dan variasi data gambar dengan menerapkan transformasi pada citra asli. Teknik ini digunakan untuk memperkuat model pembelajaran mesin dengan menyediakan data yang lebih bervariasi.

### Alat dan Bahan
1. **Software:** Python dengan library:
   - OpenCV (cv2)
   - NumPy
   - Matplotlib
2. **Dataset:** Gambar format .jpg atau .png

### Langkah-Langkah Praktikum

#### Langkah 1: Membaca dan Menampilkan Citra

**Kode Program:**
```python
import cv2
import matplotlib.pyplot as plt

# Membaca citra
img = cv2.imread('cat.jpg')

# Menampilkan citra asli
cv2.imshow("Citra asli", img)
cv2.waitKey()
```

**Hasil yang Diharapkan:**
- Citra asli ditampilkan

#### Langkah 2: Rotasi

**Kode Program:**
```python

import cv2

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

img = cv2.imread('cat.jpg')

# Rotasi 45 derajat
rotated_img = rotate_image(img, 45)

# Menampilkan hasil
cv2.imshow("Rotated image", rotated_img)
cv2.waitKey()
```

**Hasil yang Diharapkan:**
- Citra yang telah diputar sebesar 45 derajat

#### Langkah 3: Flipping

**Kode Program:**
```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Flipping horizontal
flipped_horizontally = cv2.flip(img, 1)

# Flipping vertical
flipped_vertically = cv2.flip(img, 0)

# Menampilkan hasil
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(flipped_horizontally)
plt.title('Flipping Horizontal')

plt.subplot(1, 2, 2)
plt.imshow(flipped_vertically)
plt.title('Flipping Vertical')

plt.tight_layout()
plt.show()
```

**Hasil yang Diharapkan:**
- Citra terbalik secara horizontal dan vertikal

#### Langkah 4: Penyesuaian Kecerahan

**Kode Program:**
```python
import cv2
import matplotlib.pyplot as plt

def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

img = cv2.imread("cat.jpg")
img = cv2.cvtColor(img, cv2.BGR2RGB)

# Meningkatkan kecerahan
brighter_img = adjust_brightness(img, 1.5)

# Menurunkan kecerahan
darker_img = adjust_brightness(img, 0.5)

# Menampilkan hasil
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(brighter_img)
plt.title('Kecerahan Ditingkatkan')

plt.subplot(1, 2, 2)
plt.imshow(darker_img)
plt.title('Kecerahan Diturunkan')

plt.tight_layout()
plt.show()
```

**Hasil yang Diharapkan:**
- Citra dengan kecerahan yang ditingkatkan dan diturunkan

#### Langkah 5: Penambahan Noise

**Kode Program:**
```python
import cv2
import numpy as np

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

img = cv2.imread("cat.jpg")

# Menambahkan noise
noisy_img = add_noise(img)

# Menampilkan hasil
cv2.imshow("Citra dengan noise", noisy_img)
cv2.waitKey()
```

**Hasil yang Diharapkan:**
- Citra dengan tambahan noise acak

#### Langkah 6: Scaling

**Kode Program:**
```python
import cv2

def scale_image(image, scale_factor):
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height))

img = cv2.imread("cat.jpg")

# Scaling (50%)
scaled_img = scale_image(img, 0.5)

# Menampilkan hasil
cv2.imshow("Citra dengan skala 50%", scaled_img)
cv2.waitKey()
```

**Hasil yang Diharapkan:**
- Citra dengan ukuran yang diperbesar atau diperkecil

### Tugas untuk Mahasiswa
1. Terapkan kombinasi dari beberapa augmentasi (misalnya, rotasi + flipping)
2. Gunakan citra yang berbeda dan analisis bagaimana augmentasi memengaruhi tampilan citra
3. Simpan hasil augmentasi dalam format gambar baru

---

## Bagian 3: Operasi Filter pada Citra

### Tujuan
- Memahami dan mengimplementasikan operasi filter pada citra menggunakan teknik smoothing, sharpening, dan deteksi tepi
- Memahami konsep dan implementasi gamma correction untuk penyesuaian kecerahan citra

### Alat dan Bahan
1. Komputer dengan Python (atau software pemrograman lain) terinstal
2. Library pendukung: OpenCV (cv2), NumPy, dan Matplotlib
3. Dataset citra yang sudah disediakan (atau gunakan citra apa saja)

### Pendahuluan

#### 1. Smoothing
- Smoothing digunakan untuk mengurangi noise dalam citra
- Filter smoothing yang umum digunakan adalah Gaussian Blur dan Average Blur

#### 2. Sharpening
- Sharpening digunakan untuk meningkatkan detail dalam citra dengan menonjolkan tepi

#### 3. Deteksi Tepi
- Deteksi tepi digunakan untuk menyoroti kontur objek dalam citra
- Algoritma umum: Sobel, Prewitt, atau Canny Edge Detection

#### 4. Gamma Correction
- Teknik untuk menyesuaikan kecerahan citra dengan memperbaiki intensitas piksel menggunakan fungsi eksponensial

### Langkah Praktikum

#### 1. Filter untuk Smoothing

**Teori:** Smoothing adalah teknik konvolusi menggunakan kernel rata-rata atau kernel Gaussian.

**Kode Praktik:**
```python
import cv2
from matplotlib import pyplot as plt

# Load citra
image = cv2.imread('image.jpg')  # Ganti 'image.jpg' dengan path citra Anda
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Smoothing dengan Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Smoothing dengan Average Blur
average_blur = cv2.blur(image, (5, 5))

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(gaussian_blur), plt.title('Gaussian Blur')
plt.subplot(1, 3, 3), plt.imshow(average_blur), plt.title('Average Blur')
plt.tight_layout()
plt.show()
```

#### 2. Filter untuk Sharpening

**Teori:** Menggunakan kernel dengan nilai negatif di sekitar pusat untuk meningkatkan kontras tepi.

**Kode Praktik:**
```python
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load citra
image = cv2.imread('cat.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Kernel untuk sharpening
kernel_sharpening = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])

sharpened = cv2.filter2D(image, -1, kernel_sharpening)

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(sharpened), plt.title('Sharpened')
plt.tight_layout()
plt.show()
```

#### 3. Filter untuk Deteksi Tepi

**Teori:** Menggunakan algoritma Sobel atau Canny untuk mendeteksi perubahan mendadak dalam intensitas piksel.

**Kode Praktik:**
```python
import cv2
from matplotlib import pyplot as plt

# Load citra
image = cv2.imread('cat.jpg')  # Ganti 'image.jpg' dengan path citra Anda
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Deteksi tepi dengan Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradien horizontal
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradien vertikal
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Deteksi tepi dengan Canny
edges = cv2.Canny(image, 100, 200)

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel')
plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
plt.tight_layout()
plt.show()
```

#### 4. Cara Kerja Gamma Correction

**Teori:** Gamma correction menyesuaikan intensitas piksel dengan persamaan:

I_output = I_input^γ

- Nilai gamma > 1: Menggelapkan citra
- Nilai gamma < 1: Mencerahkan citra

**Kode Praktik:**
```python
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load citra
image = cv2.imread('cat.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Fungsi untuk gamma correction
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Gamma correction
gamma_bright = gamma_correction(image, 2.0)
gamma_dark = gamma_correction(image, 0.5)

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(gamma_dark), plt.title('Gamma Dark (γ=0.5)')
plt.subplot(1, 3, 3), plt.imshow(gamma_bright), plt.title('Gamma Bright (γ=2.0)')
plt.tight_layout()
plt.show()
```

### Tugas dan Pertanyaan
1. Uji berbagai ukuran kernel untuk smoothing dan amati hasilnya
2. Buat kernel sharpening sendiri dengan kombinasi nilai yang berbeda
3. Gunakan parameter threshold yang berbeda pada Canny Edge Detection
4. Coba nilai gamma lainnya (0.3, 1.2, 2.5) dan amati perubahan citra

---

## Kesimpulan

Modul praktikum ini memberikan pemahaman komprehensif tentang pengolahan citra digital, mulai dari operasi dasar seperti konversi warna dan pemisahan channel, teknik augmentasi data untuk meningkatkan variasi dataset, hingga operasi filtering untuk enhancement dan analisis citra. Dengan menguasai teknik-teknik ini, mahasiswa dapat memahami dasar-dasar computer vision dan siap untuk mengimplementasikan aplikasi yang lebih kompleks dalam bidang pengolahan citra dan pembelajaran mesin.
