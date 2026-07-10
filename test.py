from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

# 1. LOAD DATASET AND TRAINING DATA USING RAW STRINGS
with open(r"C:\Users\HP\OneDrive\Desktop\project\data\faces_data.pkl", 'rb') as f:
    FACES = pickle.load(f)  
with open(r"C:\Users\HP\OneDrive\Desktop\project\data\names.pkl", 'rb') as f:
    LABELS = pickle.load(f)  

# Ensure list slices match array count integers
if len(FACES) != len(LABELS):
    LABELS = LABELS[:len(FACES)]  

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)
print("Model training complete. Percentage confidence system activated.")

# 2. CAMERA AND BASE WINDOW LAYOUT SETUP
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the background image safely
bg_path = r"C:\Users\HP\OneDrive\Desktop\project\backgroung.png"
image_background = cv2.imread(bg_path)

if image_background is None:
    image_background = np.zeros((720, 1280, 3), dtype=np.uint8)
    image_background[:] = (30, 25, 22) 
else:
    image_background = cv2.resize(image_background, (1280, 720)) 

COL_NAMES = ['NAME', 'TIME']
attendance_dir = r"C:\Users\HP\OneDrive\Desktop\project\Attendance"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

current_name = "Scanning..."
total_present_count = 0  
recent_logs = []  

# TRACKING DICTIONARY FOR COOL-DOWN TIMERS
last_logged_time = {}
COOLDOWN_SECONDS = 15  

# 3. LIVE RUNTIME LOOP
while True:
    ret, frame = video.read()
    if not ret:
        continue

    h_orig, w_orig, _ = frame.shape
    min_dim = min(h_orig, w_orig)
    start_x = (w_orig - min_dim) // 2
    start_y = (h_orig - min_dim) // 2
    frame_square = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    frame_final = cv2.resize(frame_square, (450, 450))
    gray = cv2.cvtColor(frame_final, cv2.COLOR_BGR2GRAY)
    
    # Highly sensitive face stabilizer configurations
    faces_rects = facedetect.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    current_epoch_time = time.time()
    date = datetime.fromtimestamp(current_epoch_time).strftime("%d-%m-%y")
    timestamp = datetime.fromtimestamp(current_epoch_time).strftime("%H:%M:%S")
    attendance_file = os.path.join(attendance_dir, f"Attendance_{date}.csv")

    # REAL-TIME EXCEL LOG COUNTER SCANNER
    if os.path.isfile(attendance_file):
        with open(attendance_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            total_present_count = max(0, len(rows) - 1) 
    else:
        total_present_count = 0

    if len(faces_rects) == 0:
        current_name = "No Face Detected"
        status_color = (130, 130, 130) 
    else:
        for (x, y, w, h) in faces_rects:
            crop_face = frame_final[y:y+h, x:x+w, :] 
            resize_img = cv2.resize(crop_face, (50, 50)).flatten().reshape(1, -1)

            # Use prediction probability scores (0.0 to 1.0) instead of raw distance numbers
            probabilities = knn.predict_proba(resize_img)
            max_probability = np.max(probabilities) 
            
            # Convert decimal score to a clean percentage out of 100%
            confidence_percentage = int(max_probability * 100)

            # STABILITY RULE: If the match confidence is greater than 75%, it's verified
            if confidence_percentage >= 100: 
                prediction = knn.predict(resize_img)
                # FIXED PERMANENTLY: Access index [0] to extract the clean text name and drop brackets
                output = str(prediction[0]).strip()
                current_name = f"Verified: {output} ({confidence_percentage}%)"
                status_color = (0, 220, 0) 
                is_unknown = False
            else:
                current_name = "Unknown Variant"
                status_color = (0, 0, 220) 
                is_unknown = True

            cv2.rectangle(frame_final, (x, y), (x+w, y+h), status_color, 2)

            if not is_unknown:
                attendance = [output, timestamp]
                exist = os.path.isfile(attendance_file)
                
                can_log = True
                if output in last_logged_time:
                    time_passed = current_epoch_time - last_logged_time[output]
                    if time_passed < COOLDOWN_SECONDS:
                        can_log = False  

                if can_log:
                    if exist:
                        with open(attendance_file, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(attendance)
                    else:
                        with open(attendance_file, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(COL_NAMES)
                            writer.writerow(attendance)
                    
                    last_logged_time[output] = current_epoch_time
                    print(f"Logged entry to CSV: {output} at {timestamp}")
                    
                    recent_logs.insert(0, f"{output} - {timestamp}")
                    if len(recent_logs) > 5:
                        recent_logs.pop()

    display_canvas = image_background.copy()
    cv2.rectangle(frame_final, (0, 0), (450, 450), status_color, 4)
    display_canvas[160:160+450, 415:415+450] = frame_final
    
    # FIXED PERMANENTLY: Extracting index [0][0] to target only the single width integer
    text_size = cv2.getTextSize(current_name, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
    text_width = text_size[0][0]
    text_x = (1280 - text_width) // 2 
    cv2.putText(display_canvas, current_name, (text_x, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 2, cv2.LINE_AA)

    # PRINT THE LOG COUNTER ON THE SCREEN
    counter_text = f"Total Present: {total_present_count}"
    cv2.putText(display_canvas, counter_text, (920, 650), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # PRINT THE ROLLING ENTRY LOG LIST ON THE RIGHT
    cv2.putText(display_canvas, "Recent Activity:", (920, 200), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    y_offset = 255
    for log in recent_logs:
        cv2.putText(display_canvas, f"> {log}", (920, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset += 40

    cv2.imshow("Face Recognition Attendance System", display_canvas)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
