import sys
import argparse
import os
import cv2
print(cv2.__version__)

def extractImages(pathIn,pathOut):
    print("Creating frames for "+pathIn)
    cap = cv2.VideoCapture(pathIn)
    count = 0
    c = 0
    success = True
    while success:
        success,image = cap.read()
        #print('read a new frame:',success)
        if count%30 == 0 :
             try:
                 cv2.imwrite(pathOut + '/frame%d.jpg'%c,image)
             except Exception as e:
                 print("")
             c+=1
        count+=1

    n = len(os.listdir(pathOut))
    for i in range(0,7):
        os.remove(pathOut+"\\frame%d.jpg" % i)
        os.remove(pathOut+"\\frame%d.jpg" % (n-i-1))

    print("Done")

if __name__=="__main__":
    parent = "./videos"
    videos = os.listdir(parent)
    for video in videos:
        pathIn = os.path.join(parent,video)
        pathOut = os.path.join("./frames",video.split('.')[0])
        try:
            os.mkdir(pathOut)
            extractImages(pathIn, pathOut)
        except Exception as e:
            print("Already Done") 
