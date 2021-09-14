# Python program to explain os.rename() method
from PIL import Image
# importing os module
import os
path = "./data"
for s in os.listdir(path):
    d = os.path.join(path,s)
    for i in os.listdir(d):
        # Source file path
        source = os.path.join(d,i)
        #print(source)
        ex = i.split('.')
        #print(ex)
        i = ex[0]+".jpeg"
        # destination file path
        dest = os.path.join(d,i)
        #print(dest)
        try:
            im = Image.open(source)
            rgb_im = im.convert('RGB')
            rgb_im.save(dest)
        except:
            os.remove(source)
        # Now rename the source path
        # to destination path
        # using os.rename() method
        #os.rename(source, dest)
print("Source path renamed to destination path successfully.")
