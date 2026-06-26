from ultralytics import YOLO
import cv2
import os

VIDEO_DATASET_DIR = "Dataset/Training"  
CNN_DATASET_DIR = "Dataset/CNN_Images"            
model_yolo = YOLO('YOLO/best.pt')

if not os.path.exists(CNN_DATASET_DIR):
    os.makedirs(CNN_DATASET_DIR)

for label in os.listdir(VIDEO_DATASET_DIR):
    class_video_dir = os.path.join(VIDEO_DATASET_DIR, label)
    
    if not os.path.isdir(class_video_dir):
        continue
    
    class_output_dir = os.path.join(CNN_DATASET_DIR, label)
    os.makedirs(class_output_dir, exist_ok=True)
    
    img_counter = 0
    
    for video_name in os.listdir(class_video_dir):
        video_path = os.path.join(class_video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        
        while True:
            success, img = cap.read()
            if not success:
                break
                
            results = model_yolo(img, stream=True, conf=0.5, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if y2 > y1 and x2 > x1:
                        
                        hand_crop = img[y1:y2, x1:x2]
                        
                        img_name = f"{label}_{img_counter}.jpg"
                        cv2.imwrite(os.path.join(class_output_dir, img_name), hand_crop)
                        img_counter += 1
                        
        cap.release()
print("Selesai mengekstrak dataset gambar untuk CNN!")