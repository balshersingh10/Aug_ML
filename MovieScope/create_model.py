from config.global_parameters import default_model_name
from utils import load_pkl,load_pkl_gzip,dump_pkl
import numpy as np
from model_utils import spatial_model



def train_classifier(genres=['romance', 'horror', 'action'], model_name=default_model_name):

    """Gather data for selected genres"""
    trainingData = []
    trainingLabels = []
#    num_of_random_frames = 75
    num_of_classes = len(genres)
    print ("Number of classes:",str(num_of_classes))
    for genreIndex, genre in enumerate(genres):
        print ("Looking for pickle file: data/{0}.p".format(genre))
        try:
            genreFeatures = load_pkl_gzip("./data/"+genre+"_ultimate_"+default_model_name+"_optimized.p")
            genreFeatures = np.array([np.array(f) for f in genreFeatures]) # numpy hack
        except Exception as e:
            print(e)
            return
        print ("OK.")
        for videoFeatures in genreFeatures:
            #"""to get all frames from a video -- hacky"""
            randomIndices = range(len(videoFeatures))
            selectedFeatures = np.array(videoFeatures[randomIndices])
            for feature in selectedFeatures:
                trainingData.append(feature)
                trainingLabels.append([genreIndex])
    trainingData = np.array(trainingData)
    dump_pkl(trainingData,"training_data_700")
    trainingLabels = np.array(trainingLabels)
    dump_pkl(trainingLabels,"training_labels_700")
    print (trainingData.shape)
    print (trainingLabels.shape)
# #    trainingLabels = to_categorical(trainingLabels, num_of_classes)
#     print( trainingLabels)
# #    trainingLabels = trainingLabels.reshape((-1,num_of_classes))
#
#     #"""Initialize the mode"""
#     model = spatial_model(num_of_classes)
#     model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
#
#     """Start training"""
#     batch_size = 5
#     nb_epoch = 100
#
#     model.fit(trainingData, trainingLabels, batch_size=batch_size, epochs=nb_epoch)#, callbacks=[remote])
#     modelOutPath ='data/models/spatial'+model_name+'_'+str(num_of_classes)+"g_bs"+str(batch_size)+"_ep"+str(nb_epoch)+".h5"
#     model.save(modelOutPath)
#     print( "Model saved at",modelOutPath)


if __name__=="__main__":

    train_classifier(genres=['Action','Comedy','Drama','Fantasy','Horror_Mystery','Romance','Thriller'])
