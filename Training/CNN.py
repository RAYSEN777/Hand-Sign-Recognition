import numpy as np
import cv2
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_DIR = 'Dataset/CNN_Images' 
IMAGE_X, IMAGE_Y = 64, 64  

def get_num_of_classes():
    return len(glob(os.path.join(DATASET_DIR, '*')))

def load_dataset():
    images = []
    labels = []
    
    classes = sorted([os.path.basename(x) for x in glob(os.path.join(DATASET_DIR, '*'))])
    print(f"Membaca kelas: {classes}")
    
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(DATASET_DIR, class_name)        
        img_paths = glob(os.path.join(class_path, '*'))
        
        for img_path in img_paths:
            try:
                img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
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
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    filepath = "cnn_model_keras2.h5"
    
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    
    return model, callbacks_list

def train():
    print("Memuat dataset dari folder...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Dataset kosong! Periksa kembali direktori DATASET_DIR Anda.")
        return
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train = np.reshape(X_train, (X_train.shape[0], IMAGE_X, IMAGE_Y, 1))
    X_val = np.reshape(X_val, (X_val.shape[0], IMAGE_X, IMAGE_Y, 1))
    
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    
    print(f"Ukuran Data Training: {X_train.shape}")
    print(f"Ukuran Data Validasi: {X_val.shape}")

    model, callbacks_list = cnn_model()
    model.summary()
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=500, callbacks=callbacks_list)
    
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

if __name__ == "__main__":
    train()
    K.clear_session()