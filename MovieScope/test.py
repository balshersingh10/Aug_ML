import os
from utils import load_pkl
import numpy as np
import tensorflow as tf

pklName='./data/200F_VGG16/26.p'
model_path = './data/models/spatialvgg16_7g_bs5_ep100.h5'
labels = ['Action','Comedy','Drama','Fantasy','Horror_Mystery','Romance','Thriller']

p = load_pkl(pklName)
model = tf.keras.models.load_model(model_path, compile=False)
pred = model.predict(p)
for i in pred:
    print(labels[np.argmax(i)])
