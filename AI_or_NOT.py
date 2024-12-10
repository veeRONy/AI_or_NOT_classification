import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import logging

import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Параметры
IMG_SIZE = (224, 224)  # размер изображения
BATCH_SIZE = 32
EPOCHS = 45

dataset_dir = 'C:/Users/user/Desktop/labeled dataset 2/'

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Подготовка данных
def load_data(data_directory):
    images = []
    labels = []
    
    for label, folder in enumerate(['generated', 'real']):
        folder_path = os.path.join(data_directory, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                print(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(label)    
                else:
                    print("Ошибка! image is None")
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Нормализация пикселей в диапазоне [0, 1]
    images = images.astype('float32') / 255.0
    
    return images, labels

# Загрузка и разделение данных
X, y = load_data(dataset_dir)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)

# Аугментация данных
train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Создание генераторов
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)
test_generator = val_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.4), # Для уменьшения переобучения
    Dense(1, activation='sigmoid')  # Выходной слой для бинарной классификации
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Количество параметров нейронной сети:")
model.summary()

# Настройка EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Обучение модели
history = model.fit(train_generator, 
                    validation_data=val_generator, 
                    epochs=EPOCHS, 
                    callbacks=[early_stopping])


png = tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Тестовая потеря: {test_loss}, Тестовая точность: {test_accuracy}")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel("Accuracy")
plt.xlabel("epochs")
plt.title("Accuracy")
plt.grid(True)
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.title("Loss")
plt.grid(True)
plt.show()

model.save('model - '+ str(test_accuracy) + '.h5')