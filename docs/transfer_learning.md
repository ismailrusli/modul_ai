# Modul 11: Face Recognition dengan Transfer Learning

**Adapted from MNIST CNN Module**

## Tujuan

1. Memahami konsep Transfer Learning
2. Menggunakan pre-trained model untuk face recognition
3. Fine-tuning model untuk dataset custom
4. Membandingkan performa dengan training from scratch

## Persiapan

Install package yang diperlukan:

```bash
pip install tensorflow matplotlib numpy pillow
```

## Persiapan Dataset

Untuk face recognition, kita akan menggunakan struktur folder seperti ini:

```
dataset/
    train/
        person1/
            img1.jpg
            img2.jpg
            img3.jpg
        person2/
            img1.jpg
            img2.jpg
        person3/
            img1.jpg
            img2.jpg
    test/
        person1/
            img1.jpg
        person2/
            img1.jpg
        person3/
            img1.jpg
```

Anda dapat menggunakan dataset seperti:
- **LFW (Labeled Faces in the Wild)**
- **CelebA**
- **Dataset custom Anda sendiri**

**Rekomendasi**: Minimal 10-20 gambar per orang untuk hasil terbaik.

## Import Libraries

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
```

## Parameter Konfigurasi

```python
# Parameter
IMG_SIZE = 160  # MobileNetV2 menggunakan ukuran 160x160 (bisa juga 224x224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5  # Ubah sesuai jumlah orang yang ingin dikenali
```

## Data Augmentation

Data augmentation sangat penting untuk face recognition agar model lebih robust terhadap variasi pose, pencahayaan, dan kondisi berbeda.

```python
# Data augmentation untuk training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi seperti pada MNIST
    rotation_range=20,  # Rotasi hingga 20 derajat
    width_shift_range=0.2,  # Geser horizontal
    height_shift_range=0.2,  # Geser vertikal
    horizontal_flip=True,  # Flip horizontal (hati-hati untuk wajah)
    zoom_range=0.2,  # Zoom in/out
    fill_mode='nearest'
)

# Untuk test data, hanya normalisasi
test_datagen = ImageDataGenerator(rescale=1./255)
```

## Load Dataset dari Folder

Ganti `'path/to/your/dataset'` dengan path dataset Anda:

```python
train_generator = train_datagen.flow_from_directory(
    'path/to/your/dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'path/to/your/dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Update jumlah kelas berdasarkan dataset
NUM_CLASSES = len(train_generator.class_indices)
print(f"Jumlah kelas yang terdeteksi: {NUM_CLASSES}")
print(f"Nama kelas: {list(train_generator.class_indices.keys())}")
```

## Apa itu Transfer Learning?

**Transfer Learning** adalah teknik menggunakan model yang sudah dilatih pada dataset besar (seperti ImageNet) sebagai base, lalu fine-tune untuk task spesifik kita.

### Keuntungan Transfer Learning:
1. ✅ Training lebih cepat
2. ✅ Memerlukan data lebih sedikit
3. ✅ Akurasi lebih tinggi
4. ✅ Model sudah belajar fitur-fitur visual yang berguna
5. ✅ Mengurangi risiko overfitting

### Perbedaan dengan MNIST CNN:
| Aspek | MNIST CNN | Face Recognition |
|-------|-----------|------------------|
| Input | 28×28 grayscale (1 channel) | 160×160 RGB (3 channels) |
| Preprocessing | Hanya normalisasi | Normalisasi + augmentation |
| Arsitektur | CNN sederhana (2-3 layers) | Transfer learning |
| Jumlah Data | 60,000 images | Bisa mulai dari 50-100 images |
| Training Time | 5-10 epochs | 1-5 epochs (transfer learning) |

## Membuat Model dengan Transfer Learning

### Fungsi untuk Membuat Model

```python
def create_transfer_learning_model(num_classes, trainable_base=False):
    """
    Membuat model face recognition dengan transfer learning
    
    Args:
        num_classes: Jumlah identitas/orang yang akan dikenali
        trainable_base: True untuk fine-tuning, False untuk feature extraction
    
    Returns:
        Model Keras yang sudah dikompilasi
    """
    
    # Load pre-trained MobileNetV2 (trained on ImageNet)
    # include_top=False: tidak menggunakan layer klasifikasi ImageNet
    # weights='imagenet': menggunakan bobot yang sudah dilatih
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers (Transfer Learning - Feature Extraction)
    base_model.trainable = trainable_base
    
    # Jika fine-tuning, hanya latih layer atas
    if trainable_base:
        # Freeze semua layer kecuali 20 layer terakhir
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    
    # Membuat model baru dengan menambahkan custom layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Menggantikan Flatten, lebih efisien
        Dense(128, activation='relu'),
        Dropout(0.5),  # Regularisasi untuk mencegah overfitting
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    
    return model
```

## Model 1: Feature Extraction (Base Model Frozen)

Dalam pendekatan ini, kita "freeze" semua layer dari pre-trained model dan hanya melatih layer yang kita tambahkan di atas.

```python
print("=" * 50)
print("Model 1: Feature Extraction (Base Model Frozen)")
print("=" * 50)

model_feature_extraction = create_transfer_learning_model(
    num_classes=NUM_CLASSES,
    trainable_base=False
)

model_feature_extraction.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_feature_extraction.summary()
```

## Model 2: Fine-tuning (Sebagian Base Model Trainable)

Dalam pendekatan ini, kita "unfreeze" sebagian layer terakhir dari pre-trained model untuk fine-tuning.

```python
print("\n" + "=" * 50)
print("Model 2: Fine-tuning (20 Layer Terakhir Trainable)")
print("=" * 50)

model_fine_tuning = create_transfer_learning_model(
    num_classes=NUM_CLASSES,
    trainable_base=True
)

model_fine_tuning.compile(
    optimizer=Adam(learning_rate=0.0001),  # Learning rate lebih kecil
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_fine_tuning.summary()
```

**Perhatikan**: Learning rate untuk fine-tuning lebih kecil (0.0001 vs 0.001) untuk mencegah merusak bobot yang sudah dilatih.

## Training Model

### Training Model 1: Feature Extraction

```python
print("\n" + "=" * 50)
print("Training Model 1: Feature Extraction")
print("=" * 50)

history_feature_extraction = model_feature_extraction.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1
)
```

### Training Model 2: Fine-tuning

```python
print("\n" + "=" * 50)
print("Training Model 2: Fine-tuning")
print("=" * 50)

history_fine_tuning = model_fine_tuning.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1
)
```

## Visualisasi Training History

```python
def plot_training_history(history, title):
    """
    Visualisasi accuracy dan loss selama training
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Visualisasi hasil training
plot_training_history(history_feature_extraction, "Feature Extraction")
plot_training_history(history_fine_tuning, "Fine-tuning")
```

## Perbandingan dengan Model From Scratch

Untuk membandingkan keunggulan transfer learning, kita juga buat model CNN dari scratch:

```python
def create_cnn_from_scratch(num_classes):
    """
    Membuat CNN dari scratch (tanpa transfer learning)
    untuk perbandingan performa
    """
    model = Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Block 4
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Buat dan latih model from scratch
model_from_scratch = create_cnn_from_scratch(NUM_CLASSES)
model_from_scratch.summary()

history_from_scratch = model_from_scratch.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1
)

plot_training_history(history_from_scratch, "From Scratch")
```

## Prediksi untuk Satu Gambar

```python
def predict_face(model, image_path, class_names):
    """
    Prediksi identitas wajah dari satu gambar
    
    Args:
        model: Model yang sudah dilatih
        image_path: Path ke gambar yang akan diprediksi
        class_names: List nama-nama kelas/identitas
    """
    from tensorflow.keras.preprocessing import image
    
    # Load dan preprocess gambar
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Visualisasi
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_names[predicted_class]}\n"
              f"Confidence: {confidence:.2%}")
    plt.show()
    
    # Tampilkan probabilitas semua kelas
    print("\nProbabilitas untuk setiap identitas:")
    for i, prob in enumerate(predictions[0]):
        print(f"{class_names[i]}: {prob:.2%}")
    
    return predicted_class, confidence

# Contoh penggunaan
class_names = list(train_generator.class_indices.keys())
predict_face(model_fine_tuning, 'path/to/test/image.jpg', class_names)
```

## Menyimpan dan Memuat Model

### Menyimpan Model

```python
# Simpan model ke file
model_fine_tuning.save('face_recognition_model.h5')
print("Model saved to face_recognition_model.h5")

# Atau simpan dalam format SavedModel (recommended)
model_fine_tuning.save('face_recognition_model')
print("Model saved to face_recognition_model/")
```

### Memuat Model

```python
from tensorflow.keras.models import load_model

# Load model dari file .h5
loaded_model = load_model('face_recognition_model.h5')
print("Model loaded successfully")

# Atau load dari SavedModel
loaded_model = tf.keras.models.load_model('face_recognition_model')
print("Model loaded successfully")

# Sekarang model dapat digunakan untuk prediksi
# predict_face(loaded_model, 'test_image.jpg', class_names)
```

## Evaluasi Model

```python
def evaluate_model(model, test_generator, model_name):
    """
    Evaluasi model pada test set
    """
    print(f"\n{'='*50}")
    print(f"Evaluasi {model_name}")
    print('='*50)
    
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return test_loss, test_accuracy

# Evaluasi semua model
results = {}
results['Feature Extraction'] = evaluate_model(
    model_feature_extraction, test_generator, "Feature Extraction"
)
results['Fine-tuning'] = evaluate_model(
    model_fine_tuning, test_generator, "Fine-tuning"
)
results['From Scratch'] = evaluate_model(
    model_from_scratch, test_generator, "From Scratch"
)

# Tampilkan perbandingan
print("\n" + "=" * 50)
print("PERBANDINGAN HASIL")
print("=" * 50)
for model_name, (loss, accuracy) in results.items():
    print(f"{model_name:20s}: Accuracy = {accuracy:.2%}, Loss = {loss:.4f}")
```

## Tips dan Best Practices

### 1. Dataset
- Minimal **10-20 gambar per orang** untuk hasil yang baik
- Variasi **pose, pencahayaan, ekspresi**
- Gunakan **data augmentation** untuk menambah variasi
- Pastikan gambar berkualitas baik dan wajah terlihat jelas

### 2. Preprocessing
- Gunakan **face detection** (MTCNN atau dlib) untuk crop wajah
- **Alignment** wajah untuk konsistensi
- Resize ke ukuran yang konsisten (160×160 atau 224×224)
- **Normalisasi** pixel values ke range [0, 1]

### 3. Transfer Learning Strategy
- Mulai dengan **feature extraction** (freeze semua layers)
- Jika akurasi kurang memuaskan, coba **fine-tuning**
- Gunakan **learning rate kecil** (0.0001) untuk fine-tuning
- Pertimbangkan **progressive unfreezing** (unfreeze layer secara bertahap)

### 4. Regularization
- Gunakan **Dropout** (0.3-0.5) untuk mencegah overfitting
- **Data augmentation** yang sesuai untuk wajah
- **Early stopping** untuk menghentikan training saat validation loss naik
- **L2 regularization** pada Dense layers jika diperlukan

### 5. Alternative Approaches
- **Face embeddings**: FaceNet, ArcFace untuk face verification
- **Siamese Networks**: Untuk few-shot learning
- **Triplet loss**: Untuk learning face representations
- **Metric learning**: Untuk similarity-based recognition

### 6. Model Selection
Pilih pre-trained model sesuai kebutuhan:
- **MobileNetV2**: Ringan, cepat, cocok untuk mobile/edge devices
- **ResNet50**: Akurasi lebih tinggi, lebih berat
- **EfficientNet**: Balance antara akurasi dan efisiensi
- **VGGFace**: Specialized untuk face recognition

## Perbandingan: MNIST CNN vs Face Recognition

| Aspek | MNIST CNN | Face Recognition |
|-------|-----------|------------------|
| **Input Shape** | 28×28×1 (grayscale) | 160×160×3 (RGB) |
| **Preprocessing** | Normalisasi | Normalisasi + Augmentation + Face Detection |
| **Arsitektur** | CNN sederhana (2-3 conv layers) | Transfer Learning (pre-trained model) |
| **Jumlah Data** | 60,000 training images | 50-1000 images (dengan transfer learning) |
| **Kompleksitas** | Pola sederhana (garis, kurva) | Pola kompleks (tekstur, bentuk, lighting) |
| **Training Time** | 5-10 epochs | 1-5 epochs (transfer learning) |
| **Akurasi** | ~98-99% | 90-98% (tergantung data) |
| **Use Case** | Digit recognition | Face identification/verification |

## Keunggulan Transfer Learning

✅ **Training lebih cepat**: 1-5 epochs vs 10-20 epochs from scratch  
✅ **Memerlukan data lebih sedikit**: 10-50 images/class vs 100-1000 images/class  
✅ **Akurasi lebih tinggi**: 90-98% vs 70-85% from scratch  
✅ **Model lebih robust**: Generalisasi lebih baik  
✅ **Mengurangi overfitting**: Pre-trained features sudah robust  
✅ **Hemat komputasi**: Tidak perlu GPU powerful untuk training  

## Tugas

1. **Prediksi Gambar Test**
    - Lakukan prediksi untuk beberapa gambar dalam test set
    - Tampilkan gambar, prediksi, dan confidence score
    - Buat confusion matrix untuk visualisasi performa

2. **Eksperimen dengan Arsitektur**
    - Coba pre-trained model lain (ResNet50, EfficientNet)
    - Bandingkan akurasi dan training time
    - Buat tabel perbandingan hasil

3. **Data Augmentation**
    - Eksperimen dengan parameter augmentation yang berbeda
    - Lihat pengaruhnya terhadap akurasi
    - Dokumentasikan hasil terbaik

4. **Fine-tuning Strategy**
    - Coba unfreeze jumlah layer yang berbeda (10, 20, 30, all)
    - Bandingkan akurasi dan training time
    - Temukan sweet spot untuk dataset Anda

5. **Real-time Face Recognition**
    - Integrasikan model dengan webcam
    - Gunakan OpenCV untuk face detection
    - Buat aplikasi sederhana untuk real-time recognition

## Referensi dan Sumber Belajar

- [Keras Applications](https://keras.io/api/applications/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Face Recognition: From Traditional to Deep Learning](https://arxiv.org/abs/1804.06655)
