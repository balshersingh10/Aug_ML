import pickle
import gzip
import numpy

#for i in range(0,100):
pklName='./genres/test/Action/0.p'
with open(pklName,'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    print(p.shape)
