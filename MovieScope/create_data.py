import os, sys
from glob import glob
import cv2
import numpy as np
from config.global_parameters import default_model_name
from config.resources import video_resource,frame_resource
#from model_utils import get_features_batch
from utils import dump_pkl,load_pkl
#from video import get_frames


def gather_training_data(genre, model_name=default_model_name):
    """Driver function to collect frame features for a genre"""

    trainPath = os.path.join(frame_resource,'genres','train',genre)
    print(trainPath)
    videoPaths = glob(trainPath+'\*')
    genreFeatures = []
    for videoPath in videoPaths:
        videoFeatures = load_pkl(videoPath)
        print (videoFeatures.shape)
        genreFeatures.append(videoFeatures)
    outPath = genre+"_ultimate_"+model_name
    dump_pkl(genreFeatures, outPath)


if __name__=="__main__":
    from sys import argv
    genre = argv[1]
    gather_training_data(genre)
