import os
import sys
import numpy as np
import random
import cv2
from keras import models
from keras import optimizers
from keras.layers import Input
from keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
from keras.callbacks import CSVLogger
import scipy
import h5py
from args import Args
from data import denormalize4gan
from discrimination import MinibatchDiscrimination
from nets import build_discriminator, build_gen, build_enc
from autoenc import *


def sample_faces(faces):
    reals = []
    for i in range( Args.batch_sz ) :
        j = random.randrange( len(faces) )
        face = faces[ j ]
        reals.append( face )
    reals = np.array(reals)
    return reals


def binary_noise(cnt):
    noise = Args.label_noise * np.random.ranf((cnt,) + Args.noise_shape) # [0, 0.1]
    noise -= 0.05 # [-0.05, 0.05]
    noise += np.random.randint(0, 2, size=((cnt,) + Args.noise_shape))
    noise -= 0.5
    noise *= 2
    return noise


def sample_fake( gen ) :
    noise = binary_noise(Args.batch_sz)
    fakes = gen.predict(noise)
    return fakes, noise


def dump_batch(imgs, cnt, ofname):
    assert Args.batch_sz >= cnt * cnt

    rows = []
    for i in range( cnt ) :
        cols = []
        for j in range(cnt*i, cnt*i+cnt):
            cols.append( imgs[j] )
        rows.append( np.concatenate(cols, axis=1) )

    alles = np.concatenate( rows, axis=0 )
    alles = denormalize4gan( alles )
    scipy.misc.imsave( ofname, alles )


def build_networks():
    shape = (Args.sz, Args.sz, 3)

    dopt = Adam(lr=0.0002, beta_1=Args.adam_beta)
    opt  = Adam(lr=0.0001, beta_1=Args.adam_beta)

    # generator part
    gen = build_gen( shape )
    gen.compile(optimizer=opt, loss='binary_crossentropy')
    gen.summary()

    # discriminator part
    disc = build_discriminator( shape )
    disc.compile(optimizer=dopt, loss='binary_crossentropy')
    disc.summary()

    noise = Input( shape=Args.noise_shape )
    gened = gen( noise )
    result = disc( gened )
    gan = models.Model( inputs=noise, outputs=result )
    gan.compile(optimizer=opt, loss='binary_crossentropy')
    gan.summary()

    return gen, disc, gan


def load_weights(model, wf):
    try:
        model.load_weights(wf)
    except:
        print("failed to load weight, network changed or corrupt hdf5", wf) #, file=sys.stderr)
        sys.exit(1)


def train_gan( dataf ) :
    gen, disc, gan = build_networks()

    print gen.summary
    print disc.summary
    print gan.summary

    # Uncomment these, if you want to continue training from some snapshot.
    # (or load pretrained generator weights)
    #load_weights(gen, Args.genw)
    #load_weights(disc, Args.discw)

    logger = CSVLogger('loss.csv') 
    logger.on_train_begin() # initialize csv file
    with h5py.File( dataf, 'r' ) as f :
        faces = f.get( 'faces' )
        run_batches(gen, disc, gan, faces, logger, range(25000))
    logger.on_train_end()


def run_batches(gen, disc, gan, faces, logger, itr_generator):
    history = [] 
    train_disc = True
    for batch in itr_generator:
        lbl_fake = Args.label_noise * np.random.ranf(Args.batch_sz)
        lbl_real = 1 - Args.label_noise * np.random.ranf(Args.batch_sz)

        fakes, noises = sample_fake( gen )
        reals = sample_faces( faces )

        if batch % 10 == 0 :
            if len(history) > Args.history_sz:
                history.pop(0) # evict oldest
            history.append( (reals, fakes) )

        gen.trainable = False
        d_loss1 = disc.train_on_batch( reals, lbl_real )
        d_loss0 = disc.train_on_batch( fakes, lbl_fake )
        gen.trainable = True

        # pretrain train discriminator only
        if batch < 20 :
            print( batch, "d0:{} d1:{}".format( d_loss0, d_loss1 ) )
            continue

        disc.trainable = False
        g_loss = gan.train_on_batch( noises, lbl_real ) # try to trick the classifier.
        disc.trainable = True

        print( batch, "d0:{} d1:{}   g:{}".format( d_loss0, d_loss1, g_loss ) )

        # save weights every 10 batches
        if batch % 10 == 0 and batch != 0 :
            end_of_batch_task(batch, gen, disc, reals, fakes)
            row = {"d_loss0": d_loss0, "d_loss1": d_loss1, "g_loss": g_loss}
            logger.on_epoch_end(batch, row)



_bits = binary_noise(Args.batch_sz)
def end_of_batch_task(batch, gen, disc, reals, fakes):
    try :
        # Dump how the generator is doing.
        # Animation dump
        dump_batch(reals, 4, "reals.png")
        dump_batch(fakes, 4, "fakes.png") # to check how noisy the image is
        frame = gen.predict(_bits)
        animf = os.path.join(Args.anim_dir, "frame_{:08d}.png".format(int(batch/10)))
        dump_batch(frame, 4, animf)
        dump_batch(frame, 4, "frame.png")

        serial = int(batch / 10) % 10
        prefix = os.path.join(Args.snapshot_dir, str(serial) + ".")

        print("Saving weights", serial)
        gen.save_weights(prefix + Args.genw)
        disc.save_weights(prefix + Args.discw)
    except KeyboardInterrupt :
        print("Saving, don't interrupt with Ctrl+C!", serial)
        end_of_batch_task(batch, gen, disc, reals, fakes)
        raise


def generate( genw, cnt ):
    shape = (Args.sz, Args.sz, 3)
    gen = build_gen( shape )
    gen.compile(optimizer='sgd', loss='mse')
    load_weights(gen, Args.genw)

    generated = gen.predict(binary_noise(Args.batch_sz))
    generated = denormalize4gan(generated)

    for i in range(cnt):
        ofname = "{:04d}.png".format(i)
        scipy.misc.imsave( ofname, generated[i] )


def main( argv ) :
    if not os.path.exists(Args.snapshot_dir) :
        os.mkdir(Args.snapshot_dir)
    if not os.path.exists(Args.anim_dir) :
        os.mkdir(Args.anim_dir)

    # test the capability of generator network through autoencoder test.
    #train_autoenc( "data.hdf5" )

    # train GAN with inputs in data.hdf5
    train_gan( "data.hdf5" )

    # generate stuff
    #generate( "gen.hdf5", 256 )



if __name__ == '__main__':
    main(sys.argv)
