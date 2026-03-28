import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32

train_dir = "dataset/train"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

# 🔥 IMPORTANT: Print class order
print("\nClass indices:", train_data.class_indices)

# 🔥 Optional: Handle class imbalance
class_counts = train_data.classes
unique, counts = np.unique(class_counts, return_counts=True)
class_weights = dict(zip(unique, 1.0 / counts))
print("Class weights:", class_weights)

# Load pretrained model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 🔥 PARTIAL UNFREEZE (better learning)
for layer in base_model.layers[-30:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # 🔥 increased
    class_weight=class_weights
)

# 🔥 Save in modern format
model.save("model/vehicle_model.keras")

print("✅ Model training complete!")