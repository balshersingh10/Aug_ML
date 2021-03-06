import pandas as pd
from shutil import copy

df = pd.read_csv (r'./movie_metadata.csv')
df = df['genres']

data_dir = './200F_VGG16'
action_dir = './genres/Train/Action/'
comedy_dir = './genres/Train/Comedy/'
drama_dir = './genres/Train/Drama/'
fantasy_dir = './genres/Train/Fantasy'
horror_mystery_dir = './genres/Train/Horror_Mystery'
romance_dir = './genres/Train/Romance/'
thriller_dir = './genres/Train/Thriller/'

for i in range(df.shape[0]):
    file_data_dir = data_dir+'/'+str(i)+'.p'
    genre = df[i].strip().split('|')
    flag = 0
    for g in genre:
        if(g=='Action'):
            try:
                file_action_dir = action_dir+'/'+str(i)+'.p'
                #os.symlink(file_data_dir,file_action_dir)
                copy(file_data_dir,action_dir)
                print("Added to "+g)
            except Exception as e:
                print("File Missing")
        elif(g=='Comedy'):
            try:
                file_comedy_dir = comedy_dir+'/'+str(i)+'.p'
                #os.symlink(file_data_dir,file_comedy_dir)
                copy(file_data_dir,comedy_dir)
                print("Added to "+g)
            except Exception as e:
                print("File Missing")
        elif(g=='Drama'):
            try:
                file_drama_dir = drama_dir+'/'+str(i)+'.p'
                #os.symlink(file_data_dir,file_drama_dir)
                copy(file_data_dir,drama_dir)
                print("Added to "+g)
            except Exception as e:
                print("File Missing")
        elif(g=='Fantasy'):
            try:
                file_fantasy_dir = fantasy_dir+'/'+str(i)+'.p'
                #os.symlink(file_data_dir,file_drama_dir)
                copy(file_data_dir,fantasy_dir)
                print("Added to "+g)
            except Exception as e:
                print("File Missing")
        elif(flag==0 and (g=='Horror' or g=='Mystery')):
            try:
                file_horror_mystery_dir = horror_mystery_dir+'/'+str(i)+'.p'
                #os.symlink(file_data_dir,file_drama_dir)
                copy(file_data_dir,horror_mystery_dir)
                flag = 1
                print("Added to "+g)
            except Exception as e:
                print("File Missing")
        elif(g=='Romance'):
            try:
                file_romance_dir = romance_dir+'/'+str(i)+'.p'
                #os.symlink(file_data_dir,file_romance_dir)
                copy(file_data_dir,romance_dir)
                print("Added to "+g)
            except Exception as e:
                print("File Missing")
        elif(g=='Thriller'):
            try:
                file_thriller_dir = thriller_dir+'/'+str(i)+'.p'
                #os.symlink(file_data_dir,file_thriller_dir)
                copy(file_data_dir,thriller_dir)
                print("Added to "+g)
            except Exception as e:
                print("File Missing")
