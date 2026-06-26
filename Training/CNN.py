import numpy as np
import cv2
import os
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\n=========================================")
gpus = tf.config.list_physical_devices('GPU')
print("Jumlah GPU Terdeteksi:", len(gpus))
if len(gpus) > 0:
    print("Nama GPU:", gpus)
    print("STATUS: Training akan berjalan menggunakan GPU! 🚀")
else:
    print("STATUS: Tidak ada GPU terdeteksi. Training menggunakan CPU. 🐢")
print("=========================================\n")

DATASET_DIR = 'Dataset/CNN_Images_Cleaned' 
IMAGE_X, IMAGE_Y = 64, 64  

def get_num_of_classes():
    """Menghitung jumlah kelas/folder di dalam direktori dataset."""
    return len(glob(os.path.join(DATASET_DIR, '*')))

def load_dataset():
    images = []
    labels = []

    classes = sorted([os.path.basename(x) for x in glob(os.path.join(DATASET_DIR, '*'))])
    print(f"Membaca kelas: {classes}")
    
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(DATASET_DIR, class_name)        
        img_paths = glob(os.path.join(class_path, '*'))

        print(f"-> Memuat {len(img_paths)} gambar dari Kelas {class_name}...")
        
        for img_path in img_paths:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMAGE_Y, IMAGE_X))
                
                images.append(img)
                labels.append(idx) 
            except Exception as e:
                print(f"Gagal membaca {img_path}: {e}")
                
    return np.array(images), np.array(labels)

def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()    

    model.add(Conv2D(16, (2,2), input_shape=(IMAGE_X, IMAGE_Y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))

    sgd = optimizers.SGD(learning_rate=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "cnn_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    
    return model, callbacks_list

def train():
    print("Memulai proses pemuatan dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("\n[ERROR] Dataset kosong! Periksa kembali jalur DATASET_DIR Anda.")
        return
    
    print("\nMemisahkan dataset (80% Train, 20% Validasi)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = np.reshape(X_train, (X_train.shape[0], IMAGE_X, IMAGE_Y, 1))
    X_val = np.reshape(X_val, (X_val.shape[0], IMAGE_X, IMAGE_Y, 1))

    X_train = X_train / 255.0
    X_val = X_val / 255.0

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    
    print(f"Ukuran Akhir Data Training: {X_train.shape}")
    print(f"Ukuran Akhir Data Validasi: {X_val.shape}\n")

    model, callbacks_list = cnn_model()
    model.summary()

    print("\nMemulai proses melatih model (Training)...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=500, callbacks=callbacks_list)

    scores = model.evaluate(X_val, y_val, verbose=0)
    print("\n=========================================")
    print("Selesai! Model terbaik disimpan sebagai: 'cnn_model_keras2.h5'")
    print("Akurasi Validasi Akhir: %.2f%%" % (scores[1] * 100))
    print("CNN Error Rate: %.2f%%" % (100 - scores[1] * 100))
    print("=========================================")

    model.save('cnn_model_final.keras')
    print("Model versi final berhasil disimpan secara manual!")

if __name__ == "__main__":
    train()
    K.clear_session()