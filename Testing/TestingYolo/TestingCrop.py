from ultralytics import YOLO
import cv2
import os

video_path = "Dataset/TestVideo/testvid2.mp4" 
cap = cv2.VideoCapture(video_path)
model = YOLO('YOLO/best.pt') 

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 800, 600) 

while True:
    sucess, img = cap.read()
    if not sucess:
        print("Video selesai diputar atau file tidak ditemukan.")
        break
        
    results = model(img, stream=True, conf=0.5) 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            if y2 > y1 and x2 > x1:
                hand_crop = img[y1:y2, x1:x2]  
                cv2.imshow("Cropped Hand", hand_crop)
                     
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()