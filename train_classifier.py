import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Проверка доступности GPU
def check_gpu_availability():
    """
    Проверяет доступность GPU для обучения.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("[INFO] Доступные GPU:", gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    else:
        print("[INFO] GPU не доступны, используется CPU.")
        return False

# 1. Создание модели
def create_model(input_shape=(128, 128, 3), num_classes=60):
    """
    Создает модель нейронной сети для классификации изображений.
    """
    print("[ШАГ 1] Создание модели...")
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[ШАГ 1] Модель создана и скомпилирована.")
    return model

# 2. Подготовка данных
def prepare_data(train_directory, validation_directory, image_size=(128, 128), batch_size=32):
    """
    Подготавливает данные для обучения и валидации.
    """
    print("[ШАГ 2] Подготовка данных...")
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    print("[ШАГ 2] Загрузка тренировочных данных...")
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("[ШАГ 2] Загрузка валидационных данных...")
    validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print(f"[ШАГ 2] Данные загружены. Количество классов: {train_generator.num_classes}")
    return train_generator, validation_generator

# 3. Обучение модели
def train_model(train_generator, validation_generator, epochs=10):
    """
    Обучает модель на тренировочных данных.
    """
    print("[ШАГ 3] Обучение модели...")
    model = create_model()

    print(f"[ШАГ 3] Начало обучения на {epochs} эпох...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        verbose=1,  # Вывод прогресса обучения (1 — с прогресс-баром)
        workers=0,  # Отключаем многопроцессорность для Windows
        use_multiprocessing=False  # Отключаем многопроцессорность
    )

    print("[ШАГ 3] Обучение завершено.")
    return model

# 4. Основная функция для первой части
def part1_main():
    """
    Основная функция для выполнения всех шагов.
    """
    print("[НАЧАЛО РАБОТЫ] Запуск программы...")

    # Проверка доступности GPU
    check_gpu_availability()

    # Пути к данным
    train_directory = r"C:\Users\Ваня\Desktop\DataSet\Synthetic Data\Training Data"
    validation_directory = r"C:\Users\Ваня\Desktop\DataSet\Synthetic Data\Validation Data"

    print(f"[ШАГ 2] Путь к тренировочным данным: {train_directory}")
    print(f"[ШАГ 2] Путь к валидационным данным: {validation_directory}")

    # Подготовка данных
    train_generator, validation_generator = prepare_data(train_directory, validation_directory)

    # Обучение модели
    model = train_model(train_generator, validation_generator, epochs=20)

    # Сохранение модели
    print("[ШАГ 4] Сохранение модели на диск...")
    model.save("trained_model.h5")  # Сохраняем модель в файл trained_model.h5
    print("[ШАГ 4] Модель сохранена в файл 'trained_model.h5'.")

    # Возвращаем обученную модель для использования в памяти
    print("[ШАГ 4] Обученная модель готова к использованию.")
    return model


if __name__ == "__main__":
    print("[ПРОГРАММА ЗАПУЩЕНА]")
    trained_model = part1_main()
    print("[ПРОГРАММА ЗАВЕРШЕНА] Модель обучена и готова к использованию.")
