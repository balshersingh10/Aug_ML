from utils import load_pkl
import numpy as np
import tensorflow as tf
import sys
from PIL import Image

ImgName = sys.argv[1]

print("Trying to open IMAGE")
Image1 = Image.open(ImgName)
Image1 = Image1.crop((32, 32, 256, 256))
img = np.array(Image1)
arr4d = np.expand_dims(img, 0)
arr4d = arr4d/255
print("IMAGE opened successfully")

model_path = './data/models/spatialvgg16_7g_bs5_ep100.h5'
labels = ['Action','Comedy','Drama','Fantasy','Horror_Mystery','Romance','Thriller']

IMG_SIZE = (224, 224)
IN_SHAPE = (*IMG_SIZE, 3)

model = tf.keras.applications.VGG16(input_shape=IN_SHAPE, include_top=True, weights='imagenet')
pretrained_model = tf.keras.models.Model(model.input, model.layers[-2].output)

features = pretrained_model.predict(arr4d)

model = tf.keras.models.load_model(model_path, compile=False)
pred = model.predict(features)
for i in pred:
    print(labels[np.argmax(i)])
