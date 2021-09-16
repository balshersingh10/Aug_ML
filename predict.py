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
#import cv2
from PIL import Image

print(tf.config.list_physical_devices('GPU'))

batch_size = 4
IMG_SIZE = (224,224)

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

model.load_weights("/content/ckpt_9")

test_dir = './test'
labels = []

for image in os.listdir(test_dir):
    image_path = os.path.join(test_dir,image)
    # Re-evaluate the model
    Image1 = Image.open(image_path)
    Image1 = Image1.crop((32, 32, 256, 256))
    img = np.array(Image1)
    #img.shape
    arr4d = np.expand_dims(img, 0)
    #arr4d.shape
    arr4d = arr4d/255
    #array = arr4d.astype(np.uint8)
    print(image+" "+labels[np.argmax(model.predict(arr4d))])
