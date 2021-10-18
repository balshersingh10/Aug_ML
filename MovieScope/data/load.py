import pickle
import gzip
import numpy

#for i in range(0,100):
pklName='./Action_ultimate_vgg16_optimized.p'
with gzip.open(pklName, 'rb') as f:
    with open('./Action_ultimate_vgg16.p','rb') as f1:
        u = pickle._Unpickler(f)
        u1 = pickle._Unpickler(f1)
        #u.encoding = 'latin1'
        p = u.load()
        p1 = u1.load()
        for i in range(0,len(p)):
            for j in range(0,len(p[0])):
                for l in range(0,len(p[0][0])):
                    if(p[i][j][l]!=p1[i][j][l]):
                        print(i,j,l)
                        break
                #print(" ")
