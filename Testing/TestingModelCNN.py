from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
import os
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model_yolo = YOLO('Model/YOLO/best.pt')

MODEL_CNN_PATH = 'Model/CNN/cnn_cleaned.keras'
if os.path.exists(MODEL_CNN_PATH):
    model_cnn = tf.keras.models.load_model(MODEL_CNN_PATH)
    print(f"Berhasil memuat model CNN: {MODEL_CNN_PATH}")
else:
    print(f"[ERROR] Model {MODEL_CNN_PATH} tidak ditemukan! Pastikan nama file sesuai.")
    exit()

IMAGE_X, IMAGE_Y = 64, 64  

DATASET_DIR = 'Dataset/CNN_Images_Cleaned' 
LABELS = sorted([os.path.basename(x) for x in glob(os.path.join(DATASET_DIR, '*'))])
print(f"Urutan Label Kelas: {LABELS}")

video_path = "Dataset/TestVideo/testvid4.mp4" 
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("Sign Language Detection (YOLO + CNN)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign Language Detection (YOLO + CNN)", 800, 600) 

while True:
    success, img = cap.read()
    if not success:
        print("Video selesai diputar atau file video tidak ditemukan.")
        break
        
    results = model_yolo(img, stream=True, conf=0.5, verbose=False) 
    
    for r in results:
        boxes = r.boxes
        for box in boxes:    
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            if y2 > y1 and x2 > x1:
                try:                    
                    hand_crop = img[y1:y2, x1:x2]  
                    gray_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                    resized_crop = cv2.resize(gray_crop, (IMAGE_Y, IMAGE_X)) 
                    normalized_crop = resized_crop / 255.0
                    img_input = np.reshape(normalized_crop, (1, IMAGE_X, IMAGE_Y, 1))
                    
                    predictions = model_cnn.predict(img_input, verbose=0)
                    class_idx = np.argmax(predictions)      
                    predicted_label = LABELS[class_idx]     
                    confidence = predictions[0][class_idx] * 100 
                    
                    text = f"{predicted_label} ({confidence:.1f}%)"
                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (0, 255, 0), 2)
                    
                    cv2.imshow("Cropped Hand (CNN Input)", gray_crop)
                    
                except Exception as e:
                    print(f"Gagal memproses frame tangan: {e}")
                     
    cv2.imshow("Sign Language Detection (YOLO + CNN)", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Deteksi selesai.")