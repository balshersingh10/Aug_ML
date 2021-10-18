from PIL import Image
import os
path = "./Videos"
for s in os.listdir(path):
    d = os.path.join(path,s)
    for i in os.listdir(d):
        source = os.path.join(d,i)
        ex = i.split('.')
        if(ex[-1]=='jpeg'):
            continue
        else:
            i = ex[0]+".jpeg"
            dest = os.path.join(d,i)
            try:
                im = Image.open(source)
                rgb_im = im.convert('RGB')
                rgb_im.save(dest)
                os.remove(source)
            except:
                os.remove(source)
print("Source path renamed to destination path successfully.")
