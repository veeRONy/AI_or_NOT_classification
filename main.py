import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import mediapipe as mp

# Загрузка обученной модели
model = tf.keras.models.load_model('model/model - 0.87 (1).h5')
IMG_SIZE = (224, 224)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Функция для поиска и обрезки лица
def extract_and_resize_face(image: np.array, img_size=(224, 224), scale_factor=1.5):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        h, w, _ = image.shape

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Центр лица
                x_center = xmin + width // 2
                y_center = ymin + height // 2

                # Расширяем границы
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                xmin_new = max(0, x_center - new_width // 2)
                ymin_new = max(0, y_center - new_height // 2)
                xmax_new = min(w, x_center + new_width // 2)
                ymax_new = min(h, y_center + new_height // 2)

                # Обрезаем лицо с расширением
                face_img = image[ymin_new:ymax_new, xmin_new:xmax_new]

                # Преобразуем из BGR в RGB
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                return face_img_rgb, cv2.resize(face_img_rgb, img_size)

    # Если лица не обнаружены, возвращаем None
    return None, None

# Функция для обработки изображений
async def handle_photo(update: Update, context: CallbackContext) -> None:
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()

    # Преобразование изображения в массив numpy
    image = Image.open(io.BytesIO(photo_bytes)).convert('RGB')
    image_array = np.array(image)

    # Извлечение лица и его ресайз
    face_img, face_img_resized = extract_and_resize_face(image_array)

    # Проверка наличия лица
    if face_img is None:
        await update.message.reply_text('На фото не обнаружено лицо. Попробуйте другое изображение.')
        return

    # Отправка вырезанного лица пользователю
    _, buffer = cv2.imencode('.jpg', face_img)
    await update.message.reply_photo(photo=io.BytesIO(buffer), caption="Найденное лицо")

    # Нормализация изображения и подготовка к модели
    face_img_resized = face_img_resized.astype('float32') / 255.0
    face_img_resized = np.expand_dims(face_img_resized, axis=0)

    # Получение предсказания
    prediction = model.predict(face_img_resized)[0][0]

    await update.message.reply_text(f'prediction {prediction:.2f}')
    # Ответ в зависимости от предсказания
    if prediction <= 0.5:
        p = (1 - prediction) * 100
        await update.message.reply_text(f'Это похоже на сгенерированное нейросетью с вероятностью {p:.2f}%')
    else:
        p = prediction * 100
        await update.message.reply_text(f'Это изображение похоже на реальное изображение с вероятностью {p:.2f}%')

# Функция для обработки команды /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Привет! Отправь мне фото, и я скажу, сгенерировано оно или нет.')

# Основная функция для запуска бота
def main():
    TOKEN = '8129867727:AAEBJ23g0vS2Hh1sHQUr1wuSsNnFUEA_Nxo'

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling()

if __name__ == '__main__':
    main()
