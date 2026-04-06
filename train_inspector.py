import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#SETTINGS
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5

#LOAD DATA
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'factory_data',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'factory_data',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    subset='validation'
)

#BUILD THE ROBOT BRAIN (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Output: 0 (Defective) or 1 (Good) - depends on folder order
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#TRAIN
print("Training the inspector...")
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

#SAVE
model.save('defect_inspector.h5')
print("✅ Model saved as 'defect_inspector.h5'")