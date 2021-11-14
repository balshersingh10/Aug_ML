import pandas as pd
import os
from distutils.dir_util import copy_tree

df = pd.read_csv("./data/movie_metadata.csv")

frame_path = "./frames"
dataset_path = "./dataset"

#'Action','Comedy','Drama','Fantasy','Horror_Mystery','Romance','Thriller'
frames = os.listdir(frame_path)
#frames = frames[3254:]

for video in frames:
    fromDirectory = os.path.join(frame_path,video)
    print(fromDirectory)
    genre = df['genres'][int(video)].strip().split('|')
    flag = 0
    for g in genre:
        if(g in ['Horror','Mystery'] and flag==0):
            toDirectory = os.path.join(dataset_path,'Horror_Mystery',video)
            copy_tree(fromDirectory, toDirectory)
            flag = 1
        elif(g in ['Action','Comedy','Drama','Fantasy','Romance','Thriller']):
            toDirectory = os.path.join(dataset_path,g,video)
            copy_tree(fromDirectory, toDirectory)
    print("Done")
