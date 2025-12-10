# Modul 12: Klasifikasi Citra dengan VGG16 dan 5-Fold Cross Validation

## Tujuan

- Memahami dan mengimplementasikan klasifikasi citra menggunakan VGG16
- Menerapkan 5-fold cross-validation untuk evaluasi model
- Menganalisis hasil klasifikasi menggunakan confusion matrix

## Alat dan Bahan

1. Komputer dengan Python atau Google Colab (disarankan untuk akses GPU)
2. Dataset citra yang telah disusun dengan folder per kelas. Disarankan mengambil dari [Kaggle](https://www.kaggle.com/datasets), cukup pilih 100 per label, tidak perlu diambil semuanya
3. Library yang diperlukan:
   - TensorFlow
   - Scikit-learn
   - Matplotlib
   - NumPy

## Langkah-Langkah Praktikum

### 1. Persiapan Lingkungan

Pastikan semua library terinstal. Gunakan perintah berikut untuk menginstal library yang dibutuhkan:

```bash
!pip install tensorflow scikit-learn matplotlib numpy opencv-python
```

### 2. Import Library

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
```

### 3. Load Dataset

Dataset harus berupa folder yang berisi sub-folder untuk setiap kelas. Contoh struktur:

```
dataset/
|-- class1/
|-- class2/
```

```python
# Load dataset dan preprocessing
datagen = ImageDataGenerator(rescale=1.0 / 255)
dataset_path = 'dataset/'  # Ganti dengan path dataset Anda

# Load semua gambar dan label
def load_data(dataset_path):
    images = []
    labels = []
    class_names = os.listdir(dataset_path)
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(class_idx)
    return np.array(images), np.array(labels), class_names

images, labels, class_names = load_data(dataset_path)
print(f"Loaded {len(images)} images across {len(class_names)} classes.")
```

### 4. 5-Fold Cross Validation

```python
# Konfigurasi Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hasil evaluasi
fold_accuracies = []
all_y_true = []
all_y_pred = []
```

### 5. Model VGG16 dan Training pada Setiap Fold

```python
# Loop melalui setiap fold
for fold, (train_idx, val_idx) in enumerate(kf.split(images, labels)):
    print(f"Training Fold {fold + 1}")
    
    # Split data
    X_train, X_val = images[train_idx], images[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    
    # Preprocessing
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=len(class_names))
    
    # Load model VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze pretrained layers
    
    # Tambahkan lapisan fully connected
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(class_names), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Training model
    model.fit(X_train, y_train_one_hot, validation_data=(X_val, y_val_one_hot), 
              epochs=10, batch_size=32, verbose=1)
    
    # Prediksi pada data validasi
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    
    # Evaluasi
    acc = accuracy_score(y_val, y_val_pred_classes)
    fold_accuracies.append(acc)
    all_y_true.extend(y_val)
    all_y_pred.extend(y_val_pred_classes)
    print(f"Accuracy Fold {fold + 1}: {acc}")
```

### 6. Evaluasi Keseluruhan dan Confusion Matrix

```python
# Confusion Matrix
conf_matrix = confusion_matrix(all_y_true, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Akurasi Total
print(f"Cross-Validation Accuracies: {fold_accuracies}")
print(f"Mean Accuracy: {np.mean(fold_accuracies)}")
```

## Tugas dan Pertanyaan

1. Uji pengaruh jumlah epoch terhadap hasil akurasi
2. Coba tambahkan lapisan tambahan pada model untuk melihat perubahan performa
3. Analisis distribusi kesalahan pada confusion matrix
