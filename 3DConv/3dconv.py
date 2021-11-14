import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate

batch_size = 4
IMG_SIZE = (224,224)

data_dir = sys.argv[1]
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=IMG_SIZE,
  batch_size=batch_size)

print(train_ds)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=IMG_SIZE,
  batch_size=batch_size)

print(val_ds)

print("Normalization started")
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_ds_val = val_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image))
print("Normalization done")

print("Sample dataset: ",normalized_ds.take(1))

class_names = train_ds.class_names
print(class_names)

def threeD_Conv(height, width, depth, classes):
  # initialize the model along with the input shape to be
  # "channels last" ordering
  model = Sequential()
  inputShape = (height, width, depth)
  model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(Conv2D(32, (3, 3), padding="same"))
  model.add(Activation("softmax"))
  model.add(MaxPooling2D((3, 3)))

  model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("softmax"))
  model.add(MaxPooling2D((3, 3)))

  # softmax classifier
  model.add(Flatten())
  model.add(Dense(classes))
  model.add(Activation("softmax"))
  # return the constructed network architecture
  return model

model = threeD_Conv(224,224,3,7)
model.summary()

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=60, decay_rate=0.9, staircase=False
)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=initial_learning_rate), loss=loss,metrics='accuracy')

import datetime
checkpoint_dir = './checkpoints/threeD_Conv_set1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
logdir = os.path.join("checkpoints/logdir_MV2_set1", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,verbose=2,
    save_best_only=True)

#uncomment below line to a particular checkpoint if you want to resume training.
#model.load_weights("ckpt_set1_a/ckpt_17")

print("Training started")
history = model.fit(normalized_ds,epochs=100,
                    validation_data=normalized_ds_val,callbacks=[checkpoint_callback,early_stopping_cb,tensorboard_callback])
