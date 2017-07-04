def train_autoenc( dataf ):

    f = h5py.File( dataf, 'r' )
    faces = f.get( 'faces' )

    opt = Adam(lr=0.001)

    shape = (Args.sz, Args.sz, 3)
    enc = build_enc( shape )
    enc.compile(optimizer=opt, loss='mse')
    enc.summary()

    # generator part
    gen = build_gen( shape )
    gen.compile(optimizer=opt, loss='mse')
    gen.summary()

    face = Input( shape=shape )
    vector = enc(face)
    recons = gen(vector)
    autoenc = models.Model( inputs=face, outputs=recons )
    autoenc.compile(optimizer=opt, loss='mse')

    epoch = 0
    while epoch < 200 :
        for i in range(10) :
            reals = sample_faces( faces  )
            fakes, noises = sample_fake( gen )
            loss = autoenc.train_on_batch( reals, reals )
            epoch += 1
            print(epoch, loss)
        fakes = autoenc.predict(reals)
        dump_batch(fakes, 4, "fakes.png")
        dump_batch(reals, 4, "reals.png")
    gen.save_weights(Args.genw)
    enc.save_weights(Args.discw)
    print("Saved", Args.genw, Args.discw)



