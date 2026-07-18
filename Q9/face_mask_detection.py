# %%
import tensorflow as tf
import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
# %% [markdown]
# 
# #### **1. Dataset Preprocessing**
# 
# %%
dataset_path = "face_mask"

IMG_SIZE = (224,224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)
# %% [markdown]
# #### **2.Load Pretrained MobileNetV2 and Add Classification Layers**
# %%
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)
# %% [markdown]
# #### **3.Compile and Train**
# %%
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)
# %% [markdown]
# #### **4.Evaluation and Inference Speed**
# %%
predictions = model.predict(validation_generator)

predictions = (predictions > 0.5).astype(int)

y_true = validation_generator.classes

accuracy = accuracy_score(y_true,predictions)
precision = precision_score(y_true,predictions)
recall = recall_score(y_true,predictions)

print("\nEvaluation Results")
print("----------------------------")
print("Accuracy :",accuracy)
print("Precision:",precision)
print("Recall   :",recall)

print(classification_report(y_true,predictions))

sample = np.expand_dims(
    next(iter(validation_generator))[0][0],
    axis=0
)

runs = 100

start = time.time()

for i in range(runs):
    model.predict(sample, verbose=0)

end = time.time()

avg_time = (end-start)/runs

print("\nAverage Inference Time")
print(avg_time,"seconds/image")

print("FPS =",1/avg_time)

model.save("FaceMask_MobileNetV2.keras")