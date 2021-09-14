import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5"
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
#from keras.utils import np_utils

print(tf.config.list_physical_devices('GPU'))

batch_size = 4
IMG_SIZE = (224,224)


data_dir = './data'
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

print("Normalization done")

print("Sample dataset: ",normalized_ds.take(1))

class_names = train_ds.class_names
print(class_names)

# Create Custom model from the pre-trained model VGG16
IMG_SHAPE = IMG_SIZE + (3,)
model_vgg16_conv = VGG16(input_shape=IMG_SHAPE, pooling='avg',weights='imagenet', include_top=False)
X = model_vgg16_conv.layers[-1].output
X = Dense(512, activation='relu', input_dim = (512,))(X)
X = Dropout(0.1)(X)
X = Dense(256, activation='relu')(X)
X = Dense(128, activation='relu')(X)
X = BatchNormalization()(X)
X = Dense(64, activation='relu')(X)
X = Dense(5, activation='softmax')(X)
model = Model(model_vgg16_conv.layers[0].output,X)
print(model.summary())

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=50, decay_rate=0.96, staircase=False
)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=initial_learning_rate), loss=loss,metrics='accuracy')

import datetime
checkpoint_dir = 'ckpt_set1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
logdir = os.path.join("logdir_set1", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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
history = model.fit(normalized_ds,epochs=30,
                    validation_data=normalized_ds_val,callbacks=[checkpoint_callback,early_stopping_cb,tensorboard_callback])

#model.load_weights("/content/ckpt_9")
