import os
import cv2


def extractImages(pathIn,pathOut):
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


extractImages("./videosss/102.mp4","./test")
