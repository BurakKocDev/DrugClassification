# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# %% Load data
# Full path to your dataset
dataset = r"C:\Users\ASUS\Desktop\veri_seti_1\3_kodlar_verisetleri\5_SagliktaDerinOgrenmeUygulamalari\Drug Vision\Data Combined"
image_dir = Path(dataset)

# Collect all image files
filepaths = list(image_dir.rglob('*.jpg')) + \
            list(image_dir.rglob('*.png')) + \
            list(image_dir.rglob('*.jpeg'))
# Extract labels as parent directory names
labels = [path.parent.name for path in filepaths]

# Create DataFrame
data = {
    'filepath': [str(p) for p in filepaths],
    'label': labels
}
image_df = pd.DataFrame(data)
print(image_df.head())

# %% Data Visualization
# Plot a random 5x5 grid of images with titles
random_indices = np.random.choice(len(image_df), size=25, replace=False)
fig, axes = plt.subplots(5, 5, figsize=(11,11))
for idx, ax in zip(random_indices, axes.flatten()):
    img = plt.imread(image_df.loc[idx, 'filepath'])
    ax.imshow(img)
    ax.set_title(image_df.loc[idx, 'label'])
    ax.axis('off')
plt.tight_layout()

# %% Train-test split
train_df, test_df = train_test_split(
    image_df, test_size=0.2, shuffle=True, random_state=42
)

# %% Data generators with MobileNetV2 preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath', y_col='label',
    target_size=(224,224), color_mode='rgb',
    class_mode='categorical', batch_size=64,
    shuffle=True, seed=42, subset='training'
)
val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath', y_col='label',
    target_size=(224,224), color_mode='rgb',
    class_mode='categorical', batch_size=64,
    shuffle=True, seed=42, subset='validation'
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath', y_col='label',
    target_size=(224,224), color_mode='rgb',
    class_mode='categorical', batch_size=64,
    shuffle=False
)

# %% Transfer learning model
pretrained_model = MobileNetV2(
    input_shape=(224,224,3), include_top=False,
    weights='imagenet', pooling='avg'
)
pretrained_model.trainable = False

# Build classification head
num_classes = len(train_generator.class_indices)
inputs = pretrained_model.input
x = pretrained_model.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = ModelCheckpoint(
    filepath='model_checkpoint.weights.h5',  # must end with .weights.h5
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True
)
earlystop_cb = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# %% Training
ehistory = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint_cb, earlystop_cb]
)

# %% Evaluation
loss, acc = model.evaluate(test_generator)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {acc:.4%}")

# %% Plot training history
epochs_range = range(1, len(ehistory.history['accuracy']) + 1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, ehistory.history['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(epochs_range, ehistory.history['val_accuracy'], 'r^-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(epochs_range, ehistory.history['loss'], 'bo-', label='Training Loss')
plt.plot(epochs_range, ehistory.history['val_loss'], 'r^-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %% Predictions and classification report
pred_probs = model.predict(test_generator)
pred_indices = np.argmax(pred_probs, axis=1)
pred_labels = [
    list(train_generator.class_indices.keys())[i]
    for i in pred_indices
]

y_true = test_df['label'].values
print(classification_report(y_true, pred_labels))
































