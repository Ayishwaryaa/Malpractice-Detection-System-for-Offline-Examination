import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input,Dropout, BatchNormalization# type: ignore
from tensorflow.keras.utils import Sequence# type: ignore

# Custom Dataset Class
class CheatingDataset(Sequence):
    def __init__(self, annotation_csv, image_folder, batch_size=32, shuffle=True):
        self.df = pd.read_csv(annotation_csv)
        self.df.dropna(subset=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'], inplace=True)
        self.df['filename'] = self.df['filename'].astype(str).str.strip()
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.df.iloc[batch_indexes]

        images = []
        labels = []

        for _, row in batch_data.iterrows():
            try:
                img_path = os.path.normpath(os.path.join(self.image_folder, row['filename']))
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                cropped = img[ymin:ymax, xmin:xmax]
                if cropped.size == 0:
                    continue

                resized = cv2.resize(cropped, (224, 224))
                preprocessed = preprocess_input(resized)
                images.append(preprocessed)
                labels.append(int(row['class']))
            except Exception as e:
                print(f" Error reading row: {e}")
                continue

        if len(images) == 0:
            # Prevents from crashing when batch is empty
            return np.zeros((1, 224, 224, 3)), np.zeros((1,))

        return np.array(images), np.array(labels)

#Paths
base_path = r'E:/my_files/final_year_project/image_dataset/ds3-tf'

train_csv = os.path.normpath(os.path.join(base_path, 'train', '_annotations_new.csv'))
train_img = os.path.normpath(os.path.join(base_path, 'train', 'images'))

val_csv = os.path.normpath(os.path.join(base_path, 'valid', '_annotations_new.csv'))
val_img = os.path.normpath(os.path.join(base_path, 'valid', 'images'))

test_csv = os.path.normpath(os.path.join(base_path, 'test', '_annotations_new.csv'))
test_img = os.path.normpath(os.path.join(base_path, 'test', 'images'))

# Create Data Generators
train_gen = CheatingDataset(train_csv, train_img)
val_gen = CheatingDataset(val_csv, val_img, shuffle=False)
test_gen = CheatingDataset(test_csv, test_img, shuffle=False)

# Build Model
def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Add custom head
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Fine-tune last 40 layers of ResNet50
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train Model
model = build_model()

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=1
)

# Save Model
model.save("model.keras")

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
