""" Contains all useful functions """

#from cPickle import load, dump
import pickle
import gzip
import numpy
from keras.models import load_model

from config.resources import model_resource

def sliding_window(image, windowSize, horizontalStride=4, verticalStride=4):

    """
    Runs a sliding window generator across the image
        Inputs:
            image: input image/frame
            windowSize: Tuple (width, height)
            stride: Window step size horizontal & vertical
        Output:
            Generator object with coordinates and the window cropped
    """

    for y in xrange(0, image.shape[0], verticalStride):
        for x in xrange(0, image.shape[1], horizontalStride):
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

'''
def load_pkl(pklName, verbose=True):
    if verbose:
        print ("Loading data from data/{0}.p".format(pklName))
    with open(pklName, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p
'''
def load_pkl(pklName, verbose=True):
    if verbose:
        print ("Loading data from data/{0}".format(pklName))
    with open(pklName, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p
def load_pkl_gzip(pklName, verbose=True):
    if verbose:
        print ("Loading data from data/{0}.p".format(pklName))
    with gzip.open(pklName, 'rb') as f:
        u = pickle._Unpickler(f)
        #u.encoding = 'latin1'
        p = u.load()
        return p

def dump_pkl(data, pklName, verbose = True):

    if verbose:
        print ("Dumping data into",pklName)
    pickle.dump(data, open('data/'+pklName+'.p', 'wb'))

def load_moviescope_model(modelName, verbose=True):

    if modelName.find('h5')==-1:
        modelName+=".h5"
    if verbose:
        print ("Loading model:",modelName)
    model = load_model(model_resource+modelName)
    return model
