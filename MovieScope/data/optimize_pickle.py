import gzip, pickle, pickletools
filepath = "./Thriller_ultimate_vgg16.p"

with open(filepath, 'rb') as f:
    u = pickle._Unpickler(f)
    #u.encoding = 'latin1'
    p = u.load()
    #print(len(p),len(p[0]),len(p[0][0]))
    new_filepath = "./Thriller_ultimate_vgg16_optimized.p"
    with gzip.open(new_filepath, "wb") as f:
        pickled = pickle.dumps(p)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)
