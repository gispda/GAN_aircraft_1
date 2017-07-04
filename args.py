
class Args :
    # dataset size... Use positive number to sample subset of the full dataset.
    dataset_sz = -1

    # Archive outputs of training here for animating later.
    anim_dir = "ac"

    # images size we will work on. (sz, sz, 3)
    sz = 128
    
    # alpha, used by leaky relu of D and G networks.
    alpha_D = 0.2
    alpha_G = 0.2

    # batch size, during training.
    batch_sz = 32

    # Length of the noise vector to generate the faces from.
    # Latent space z
    noise_shape = (1, 1, 400)

    # GAN training can be ruined any moment if not careful.
    # Archive some snapshots in this directory.
    snapshot_dir = "./snapshots"

    # dropout probability
    dropout = 0.4

    # noisy label magnitude
    label_noise = 0.1

    # history to keep. Slower training but higher quality.
    history_sz = 8

    genw = "gen.hdf5"
    discw = "disc.hdf5"

    kernel_initializer = 'glorot_uniform'

    # Since DCGAN paper, everybody uses 0.5 and for me, it works the best too.
    # I tried 0.9, 0.1.
    adam_beta = 0.5

    # BatchNormalization matters too.
    bn_momentum = 0.3  # 0.5
