import os
import subprocess
import random
import time
from music21 import stream, note, clef, environment
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import shutil
from PIL import Image, ImageEnhance, ImageOps
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка MuseScore 4 для music21
env = environment.Environment()
env['musicxmlPath'] = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
env['musescoreDirectPNGPath'] = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

# Пути к данным
SYNTHETIC_DATA_DIR = r"D:\DataSet"
TRAINING_DATA_DIR = os.path.join(SYNTHETIC_DATA_DIR, "Training Data")
VALIDATION_DATA_DIR = os.path.join(SYNTHETIC_DATA_DIR, "Validation Data")

# Список всех нот и их частот
notes = {
    "C1": 32, "C#1": 34, "D1": 36, "D#1": 38, "E1": 41, "F1": 43, "F#1": 46, "G1": 49,
    "G#1": 52, "A1": 55, "A#1": 58, "B1": 61, "C2": 65, "C#2": 69, "D2": 73, "D#2": 77,
    "E2": 82, "F2": 87, "F#2": 92, "G2": 98, "G#2": 104, "A2": 110, "A#2": 116, "B2": 123,
    "C3": 130, "C#3": 138, "D3": 146, "D#3": 155, "E3": 164, "F3": 174, "F#3": 185, "G3": 196,
    "G#3": 208, "A3": 220, "A#3": 233, "B3": 246, "C4": 261, "C#4": 277, "D4": 293, "D#4": 311,
    "E4": 329, "F4": 349, "F#4": 369, "G4": 392, "G#4": 415, "A4": 440, "A#4": 466, "B4": 493,
    "C5": 523, "C#5": 554, "D5": 587, "D#5": 622, "E5": 659, "F5": 698, "F#5": 739, "G5": 784,
    "G#5": 830, "A5": 880, "A#5": 932, "B5": 987,
}

# Загрузка модели Mask2Former и процессора
MODEL_NAME = "facebook/mask2former-swin-small-coco-instance"
processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)

# Функция для проверки наличия файла с суффиксом -1
def get_image_path(base_path: str) -> str:
    if os.path.exists(base_path):
        return base_path
    base_path_with_suffix = base_path.replace(".png", "-1.png")
    if os.path.exists(base_path_with_suffix):
        return base_path_with_suffix
    raise FileNotFoundError(f"Файл {base_path} или {base_path_with_suffix} не найден.")

# Функция для генерации нотного листа с одной нотой
def generate_music_sheet(note_name: str, output_image_path: str):
    try:
        logger.info(f"Генерация нотного листа для ноты {note_name}...")
        s = stream.Stream()
        s.append(clef.TrebleClef())
        n = note.Note(note_name)
        s.append(n)
        musicxml_path = f"temp_{note_name}_{os.getpid()}.musicxml"
        s.write('musicxml', fp=musicxml_path)

        # Удаляем старые файлы, если они существуют
        if os.path.exists(output_image_path):
            os.remove(output_image_path)
        if os.path.exists(output_image_path.replace(".png", "-1.png")):
            os.remove(output_image_path.replace(".png", "-1.png"))

        # Экспорт в PNG с высоким качеством (без --export-dpi)
        logger.info(f"Конвертация MusicXML в PNG...")
        subprocess.run([
            env['musicxmlPath'],
            musicxml_path,
            "-o", output_image_path,  # Экспорт в PNG
        ], check=True)
        time.sleep(2)  # Даем время для завершения экспорта
        if os.path.exists(musicxml_path):
            os.remove(musicxml_path)
            logger.info(f"Временный файл {musicxml_path} удалён.")

        logger.info(f"Нотный лист сохранён как {output_image_path}")
    except Exception as e:
        logger.error(f"Ошибка при генерации ноты {note_name}: {e}")

# Функция для обрезки изображения с помощью модели Mask2Former
def crop_image_with_model(image_path: str, output_path: str):
    try:
        # Проверяем наличие файла с суффиксом -1
        image_path = get_image_path(image_path)

        # Загрузка изображения
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")

        # Прогнозирование маски
        with torch.no_grad():
            outputs = model(**inputs)
        mask = outputs.class_queries_logits.argmax(dim=1).squeeze().cpu().numpy()

        # Создание маски для нот (предположим, что нота соответствует классу 1)
        note_mask = (mask == 1).astype(np.uint8) * 255

        # Применение маски к изображению
        masked_image = Image.fromarray(note_mask)
        masked_image.save(output_path)
        logger.info(f"Изображение {image_path} обрезано и сохранено как {output_path}.")
    except Exception as e:
        logger.error(f"Ошибка при обрезке изображения {image_path}: {e}")

# Функция для аугментации изображения
def augment_image(image_path: str, output_path: str, augmentation_level: int):
    try:
        # Проверяем наличие файла с суффиксом -1
        image_path = get_image_path(image_path)

        image = Image.open(image_path)
        if augmentation_level == 1:  # Минимальная аугментация
            angle = random.uniform(-2, 2)
            x_shift = random.randint(-2, 2)
            y_shift = random.randint(-2, 2)
        elif augmentation_level == 2:  # Умеренная аугментация
            angle = random.uniform(-5, 5)
            x_shift = random.randint(-5, 5)
            y_shift = random.randint(-5, 5)
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)

        # Применение аугментации
        image = image.rotate(angle)
        image = image.transform(image.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))
        if augmentation_level == 2:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        image.save(output_path)
        logger.info(f"Аугментированное изображение (уровень {augmentation_level}) сохранено как {output_path}")
    except Exception as e:
        logger.error(f"Ошибка при аугментации изображения {image_path}: {e}")

# Функция для генерации изображений для одной ноты
def generate_note_images(note_name: str, num_images: int):
    class_dir = os.path.join(SYNTHETIC_DATA_DIR, note_name)
    os.makedirs(class_dir, exist_ok=True)
    for i in range(num_images):
        output_image_path = os.path.join(class_dir, f"{note_name}_{i + 1}.png")
        generate_music_sheet(note_name, output_image_path)

        # Обрезаем изображение с помощью модели
        cropped_image_path = os.path.join(class_dir, f"{note_name}_{i + 1}_cropped.png")
        crop_image_with_model(output_image_path, cropped_image_path)

        # Определяем уровень аугментации
        if i < 10:  # Первые 10 изображений: минимальная аугментация
            augmentation_level = 1
        else:  # Остальные 20 изображений: умеренная аугментация
            augmentation_level = 2

        # Применяем аугментацию к обрезанному изображению
        augment_image(cropped_image_path, cropped_image_path, augmentation_level)

    logger.info(f"Генерация изображений для ноты {note_name} завершена.")

# Основная функция для генерации данных
def main():
    num_images_per_class = 30  # 30 изображений на каждую ноту

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for note_name in notes.keys():
            future = executor.submit(generate_note_images, note_name, num_images_per_class)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Ошибка в процессе: {e}")

    logger.info("Генерация данных завершена.")

if __name__ == "__main__":
    main()
