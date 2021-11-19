import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
from PIL import Image
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model_path = './data/models/threeDConv.h5'
labels = ['Action','Comedy','Drama','Fantasy','Horror_Mystery','Romance','Thriller']

IMG_SIZE = (224, 224)
IN_SHAPE = (*IMG_SIZE, 3)

model = tf.keras.models.load_model(model_path, compile=False)

ImgDir = sys.argv[1]
Images = os.listdir(ImgDir)


for image in Images:
    ImgName = os.path.join(ImgDir,image)
    print("Trying to open IMAGE")
    Image1 = Image.open(ImgName)
    Image1 = Image1.crop((32, 32, 256, 256))
    img = np.array(Image1)
    arr4d = np.expand_dims(img, 0)
    arr4d = arr4d/255
    print("IMAGE opened successfully")
    pred = model.predict(arr4d)
    print("Image: "+image+" Prediction: "+labels[np.argmax(pred)])
