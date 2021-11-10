import gzip, pickle, pickletools
all_filepaths = ["Action_ultimate_vgg16.p","Comedy_ultimate_vgg16.p","Drama_ultimate_vgg16.p","Fantasy_ultimate_vgg16.p","Horror_Mystery_ultimate_vgg16.p","Romance_ultimate_vgg16.p","Thriller_ultimate_vgg16.p"]

for filepath in all_filepaths:
    with open(filepath, 'rb') as f:
        u = pickle._Unpickler(f)
        #u.encoding = 'latin1'
        p = u.load()
        #print(len(p),len(p[0]),len(p[0][0]))
        new_filepath = filepath[:-2] + "_optimized.p"
        with gzip.open(new_filepath, "wb") as f:
            pickled = pickle.dumps(p)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
