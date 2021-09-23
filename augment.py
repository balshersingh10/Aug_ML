from PIL import Image
import tensorflow as tf
import os

data_augmentation = tf.keras.Sequential([
     tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])

path = "./data/car_nitro"
for i in os.listdir(path):
    source = os.path.join(path,i)
    ex = i.split('.')
    i = ex[0]+"AUG"
    im = Image.open(source)
    rgb_im = im.convert('RGB')
    image = tf.expand_dims(rgb_im, 0)
    for j in range(9):
        augmented_image = data_augmentation(image)
        des = i + str(j) + ".jpeg"
        dest = os.path.join(path,des)
        tf.keras.preprocessing.image.save_img(dest,augmented_image[0])
