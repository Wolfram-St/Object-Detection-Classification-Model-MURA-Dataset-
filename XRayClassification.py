import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image, UnidentifiedImageError

def create_dataframe(dataset_dir):
    image_paths = []
    labels = []
    bad_files = []

    for body_part in os.listdir(dataset_dir):
        body_part_path = os.path.join(dataset_dir, body_part)
        if os.path.isdir(body_part_path):
            for patient in os.listdir(body_part_path):
                patient_path = os.path.join(body_part_path, patient)
                if os.path.isdir(patient_path):
                    for study in os.listdir(patient_path):
                        study_path = os.path.join(patient_path, study)
                        if os.path.isdir(study_path):

                            label = 'Abnormal' if 'positive' in study else 'Normal'

                            for image_file in os.listdir(study_path):
                                if image_file.startswith('.') or image_file.startswith('._'):
                                    continue
                                if image_file.lower().endswith('.png'):
                                    p = os.path.join(study_path, image_file)
                                    try:
                                        with Image.open(p) as im:
                                            im.verify()
                                        image_paths.append(p)
                                        labels.append(label)
                                    except (UnidentifiedImageError, OSError):
                                        bad_files.append(p)

    if bad_files:
        print(f"Skipped {len(bad_files)} unreadable images. Example: {bad_files[:5]}")

    return pd.DataFrame({'image_path': image_paths, 'label': labels})

train_dir='D:\\X-Ray\\MURA-v1.1\\train'
valid_dir='D:\\X-Ray\\MURA-v1.1\\valid'

train_df = create_dataframe(train_dir)
valid_df = create_dataframe(valid_dir)

print(f'Training samples: {len(train_df)}')
print(f'Validation samples: {len(valid_df)}')   
print(train_df.head())

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True, 
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_loader = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
)

valid_loader = val_datagen.flow_from_dataframe(
    dataframe=valid_df, 
    x_col='image_path',
    y_col='label',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False,
)

image, label = next(train_loader)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image[i])
    plt.title(f"Label: {int(label[i])}")
    plt.axis('off')

plt.show()

base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(128, activation='relu')(x)
x=Dropout(0.2)(x)
predictions=Dense(1, activation='sigmoid')(x)

model=Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()



history=model.fit(
    train_loader,
    epochs=5,
    validation_data=valid_loader
)

model.save("models/xray_mobilenetv2.h5")

plt.figure(figsize=(12, 4)) 
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.savefig('training_history.png')
plt.show()

test_images, test_labels = next(valid_loader)
predictions = model.predict(test_images)


plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i])
    
    pred_label = 'Abnormal' if predictions[i] > 0.5 else 'Normal'
    true_label = 'Abnormal' if test_labels[i] == 1 else 'Normal'

    plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
    plt.axis('off')
plt.savefig('sample_predictions.png')
plt.show()