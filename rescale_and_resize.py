import mediapipe as mp
import os
import cv2

IMG_SIZE = (224, 224)  # размер изображения

dataset_dir = 'C:/Users/user/Desktop/dataset2/'

labeled_dataset_dir = 'C:/Users/user/Desktop/labeled dataset 2/'

def crop_and_resize_image(image, scale_factor=1.5):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                      int(bboxC.width * w), int(bboxC.height * h)
                
                # Центр лица
                x_center = x + width // 2
                y_center = y + height // 2
                
                # Расширяем границы
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                xmin_new = max(0, x_center - new_width // 2)
                ymin_new = max(0, y_center - new_height // 2)
                xmax_new = min(w, x_center + new_width // 2)
                ymax_new = min(h, y_center + new_height // 2)
                
                cropped_image = image[ymin_new:ymax_new, xmin_new:xmax_new]
                
                try:
                  return cv2.resize(cropped_image, IMG_SIZE)
                except Exception as e:
                  return None        
        else:
            return None
    return cv2.resize(image, IMG_SIZE)



for label, folder in enumerate(['generated', 'real']):
    folder_path = os.path.join(dataset_dir, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                processed_image = crop_and_resize_image(image)
                if processed_image is not None:
                    new_path = os.path.join(labeled_dataset_dir, folder, filename)
                    isWritten = cv2.imwrite(new_path, processed_image)
                    if isWritten:
                        print(filename, ' ok')
                    else:
                        print(filename, ' no')
                else:
                    print("Ошибка! processed_image is None", filename)
            else:
                print("Ошибка! image is None", filename)
                
                
                
                