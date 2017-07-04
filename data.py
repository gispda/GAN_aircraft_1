import glob
import h5py
import cv2
import numpy as np
import scipy.misc
import random
from args import Args



def normalize4gan(im):
    im = im.astype(np.float32)
    im /= 127.5
    im -= 1.0 
    return im



def denormalize4gan(im):
    im += 1.0 
    im *= 127.5 
    return im.astype(np.uint8)



def make_hdf5(ofname, wildcard):
    # take only 2500 images
    pool = list(glob.glob(wildcard))[:5000]
    if Args.dataset_sz <= 0:
        fnames = pool
    else:
        fnames = []
        for i in range(Args.dataset_sz):
            # possible duplicate but don't care
            fnames.append(random.choice(pool))

    with h5py.File(ofname, "w") as f:
        faces = f.create_dataset("faces", (len(fnames), Args.sz, Args.sz, 3), dtype='f')

        for i, fname in enumerate(fnames):
            print(fname)
            im = scipy.misc.imread(fname, mode='RGB') # some have alpha channel
            im = scipy.misc.imresize(im, (Args.sz, Args.sz))
            faces[i] = normalize4gan(im)



def test(hdff):
    with h5py.File(hdff, "r") as f:
        X = f.get("faces")
        print(np.min(X[:,:,:,0]))
        print(np.max(X[:,:,:,0]))
        print(np.min(X[:,:,:,1]))
        print(np.max(X[:,:,:,1]))
        print(np.min(X[:,:,:,2]))
        print(np.max(X[:,:,:,2]))
        print("Dataset size:", len(X))
        assert np.max(X) <= 1.0
        assert np.min(X) >= -1.0



if __name__ == "__main__" :
    make_hdf5("data.hdf5", "/home/kb/Downloads/fgvc-aircraft-2013b/data/images/*.jpg")
    test("data.hdf5")
