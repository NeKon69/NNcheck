import tensorflow.keras as keras

def create_classifier_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(FREQ), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_classifier_model(train_directory, validation_directory):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_directory, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(validation_directory, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

    model = create_classifier_model()
    model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

    model.save('note_classifier_model.h5')